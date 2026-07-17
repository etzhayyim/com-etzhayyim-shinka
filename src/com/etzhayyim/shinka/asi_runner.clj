(ns com.etzhayyim.shinka.asi-runner
  "Operator-facing preflight for one Shinka ASI beat.

  This command is intentionally plan-only.  It makes the live-health gate
  observable without giving a scheduled process authority to dispatch work,
  sign a CACAO, or materialize a model/code change."
  (:require [clojure.edn :as edn]
            [cheshire.core :as json]
            [clojure.pprint :as pprint]
            [com.etzhayyim.shinka.asi :as asi])
  (:import [java.net URI]
           [java.net.http HttpClient HttpRequest HttpResponse$BodyHandlers]))

(def default-cloud-url "https://api.murakumo.cloud")
(def GiB (* 1024 1024 1024))

(defn normalize-fleet
  "Translate the public `/infer/fleet` representation into the narrow Shinka
  observation contract. Only explicitly inference-capable healthy nodes pass;
  unknown attributes never become an inferred capability."
  [snapshot]
  (let [shard-ceiling (some-> (:per-node-shard-ceiling-gb snapshot) (* GiB))]
    (mapv
     (fn [node]
       (let [caps (set (or (:caps node) (:node/can node)))
             roles (set (:roles node))
           inference? (or (contains? caps :inference)
                          (contains? caps :prompt-eval)
                          (contains? caps :host-large-model)
                          (contains? roles "llama")
                          (contains? roles "ollama"))
             mem-bytes (or (:mem-bytes node) (get-in node [:node/caps :mem-bytes])
                           (some-> (:ram-gb node) (* GiB)) 0)
             healthy? (contains? #{"ok" "healthy" "up" :ok :healthy :up true}
                                (or (:health node) (:status node)))]
         {:name (or (:name node) (:node/name node) (:id node))
          :health (if (and inference? healthy?) :healthy :down)
          :caps (if inference? #{:inference} #{})
          :mem-bytes mem-bytes
          :wired-limit-bytes shard-ceiling
          :link-gbps (or (:link-gbps node) (get-in node [:node/caps :link-gbps]))}))
     (or (:nodes snapshot) []))))

(defn fleet-summary
  "Explain the normalized capacity decision without exposing host identities.
  This is returned with every preflight so operators can distinguish a genuine
  fleet outage from a capability or memory-shape mismatch."
  [nodes]
  {:observed-nodes (count nodes)
   :healthy-inference-nodes (count (filter #(and (= :healthy (:health %))
                                                  (contains? (:caps %) :inference)) nodes))
   :healthy-names (mapv :name (filter #(= :healthy (:health %)) nodes))})

(defn fetch-fleet
  "Read the public cloud-murakumo capacity snapshot. This is the sole network
  call in the runner and is deliberately separated from planning and dispatch."
  ([] (fetch-fleet (or (System/getenv "MURAKUMO_CLOUD") default-cloud-url)))
  ([base-url]
   (let [request (-> (HttpRequest/newBuilder (URI/create (str base-url "/infer/fleet")))
                     (.header "accept" "application/json")
                     .GET .build)
         response (.send (HttpClient/newHttpClient) request
                         (HttpResponse$BodyHandlers/ofString))]
     (when-not (= 200 (.statusCode response))
       (throw (ex-info "cloud-murakumo fleet request failed"
                       {:status (.statusCode response)})))
     (json/parse-string (.body response) true))))

(defn fetch-models
  "Read the public serving-model catalog. A live ASI beat may only dispatch a
  model explicitly marked `serving`; registered or benchmark-only entries are
  not execution capacity."
  ([] (fetch-models (or (System/getenv "MURAKUMO_CLOUD") default-cloud-url)))
  ([base-url]
   (let [request (-> (HttpRequest/newBuilder (URI/create (str base-url "/infer/models")))
                     (.header "accept" "application/json")
                     .GET .build)
         response (.send (HttpClient/newHttpClient) request
                         (HttpResponse$BodyHandlers/ofString))]
     (when-not (= 200 (.statusCode response))
       (throw (ex-info "cloud-murakumo model request failed"
                       {:status (.statusCode response)})))
     (json/parse-string (.body response) true))))

(defn- gateway-models-url [endpoint]
  (if-let [[_ base] (re-matches #"(https?://.+)/v1/chat/completions" endpoint)]
    (str base "/v1/models")
    (throw (ex-info "unsupported cloud-murakumo gateway endpoint" {:endpoint endpoint}))))

(defn fetch-runtime-models
  "Read the gateway's own `/v1/models` response. The public control-plane
  catalog is not treated as sufficient evidence of what the model server has
  actually loaded."
  [endpoint]
  (let [request (-> (HttpRequest/newBuilder (URI/create (gateway-models-url endpoint)))
                    (.header "accept" "application/json")
                    .GET .build)
        response (.send (HttpClient/newHttpClient) request
                        (HttpResponse$BodyHandlers/ofString))]
    (when-not (= 200 (.statusCode response))
      (throw (ex-info "cloud-murakumo runtime model request failed"
                      {:endpoint endpoint :status (.statusCode response)})))
    (let [body (json/parse-string (.body response) true)
          entries (or (:data body) (:models body) [])]
      (->> entries
           (mapcat #(cons (or (:id %) (:model %) (:name %)) (:aliases %)))
           (filter string?)
           vec))))

(defn model-availability
  "Classify the requested model against the live catalog without guessing from
  hardware capacity. The returned public ids make an operator correction easy."
  [model catalog runtime-models]
  (let [available (mapv :id catalog)
        entry (some #(when (= (:model/id model) (:id %)) %) catalog)
        runtime-ids (get runtime-models (:endpoint entry) [])]
    (cond
      (not= "serving" (:status entry))
      {:status :not-serving :available-models available}

      (and (:endpoint entry) (not (some #{(:model/id model)} runtime-ids)))
      {:status :runtime-unverified :endpoint (:endpoint entry)
       :catalog-model entry :runtime-model-ids runtime-ids}

      :else
      {:status :serving
       :execution-mode (when (:endpoint entry) :gateway)
       :endpoint (:endpoint entry)
       :model entry})))

(defn preflight
  "Return a concise operational verdict and the complete immutable beat plan.
  `:ready?` means the caller may submit the returned enqueue effect to the
  cloud-murakumo relay; it does not authorize promotion or deployment."
  [input]
  (let [plan (asi/beat-plan input)
        dispatch (:dispatch plan)]
    {:ready? (= :ready (:status dispatch))
     :reason (or (:reason dispatch) :fleet-ready)
     :candidate-id (:candidate-id input)
     :benchmark-id (:benchmark-id input)
     :fleet (fleet-summary (:nodes input))
     :plan plan}))

(defn read-input [path]
  (edn/read-string (slurp path)))

(defn live-preflight
  "Fetch and normalize the live fleet before preflight. Failure becomes a
  deterministic blocked plan rather than an exception that a scheduler might
  accidentally retry into a stale observation. `fetch` is injectable for tests."
  ([input]
   (live-preflight
    input
    (fn []
      (let [models (fetch-models)
            endpoints (->> models (filter #(and (= "serving" (:status %)) (:endpoint %)))
                           (map :endpoint) distinct)]
        {:fleet (fetch-fleet)
         :models models
         :runtime-models (into {} (map (fn [endpoint] [endpoint (fetch-runtime-models endpoint)]) endpoints))}))))
  ([input fetch]
   (try
     (let [{:keys [fleet models runtime-models]} (fetch)]
       (preflight (assoc input
                         :nodes (normalize-fleet fleet)
                         :model-availability (model-availability (:model input) models runtime-models))))
     (catch Exception e
       (assoc (preflight (assoc input :nodes []))
              :ready? false
              :reason :fleet-observation-failed
              :observation-error (ex-message e))))))

(defn -main
  "`clojure -M -m com.etzhayyim.shinka.asi-runner beat.edn`.
  Outputs EDN and exits 0 for ready, 2 for an expected blocked preflight."
  [& args]
  (let [[live? path] (if (= "--live" (first args)) [true (second args)] [false (first args)])]
  (if-not path
    (binding [*out* *err*]
      (println "usage: bb asi-check [--live] <beat.edn>"))
    (let [result ((if live? live-preflight preflight) (read-input path))]
      (pprint/pprint result)
      (System/exit (if (:ready? result) 0 2))))))
