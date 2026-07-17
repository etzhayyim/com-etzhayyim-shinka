(ns com.etzhayyim.shinka.catalog-reconcile
  "Explicit, CACAO-gated executor for a reviewed cloud-murakumo model catalog
  reconciliation. Planning is pure; HTTP is injected and never invoked by the
  default CLI path."
  (:require [clojure.edn :as edn]
            [cheshire.core :as json]
            [clojure.pprint :as pprint])
  (:import [java.net URI]
           [java.net.http HttpClient HttpRequest HttpRequest$BodyPublishers
                          HttpResponse$BodyHandlers]))

(def default-api "https://api.murakumo.cloud")

(defn- ->catalog-record [upsert endpoint runtime-ids]
  {:id (:model/id upsert)
   :family (:model/family upsert)
   :format (name (:model/format upsert))
   :layers (:model/layers upsert)
   :params-b (:model/params-b upsert)
   :active-params-b (:model/active-params-b upsert)
   :context (:model/context upsert)
   :weight-bytes (:model/weight-bytes upsert)
   :status "serving"
   :endpoint endpoint
   :runtime-model-ids (vec runtime-ids)})

(defn reconciliation-plan
  "Turn a reviewed, member-signoff proposal into two explicit catalog writes.
  The new runtime-proven entry is written first; the stale serving entry is
  then retired. Callers may display this plan or execute it with `apply!`."
  [{:keys [payload] :as proposal}]
  (when-not (:requires-member-cacao payload)
    (throw (ex-info "reconciliation requires member CACAO" {:proposal proposal})))
  (let [{:keys [catalog-model-id recommended-upsert endpoint runtime-model-ids
                recommended-retire-serving-id]} payload]
    (when-not (and catalog-model-id recommended-upsert endpoint recommended-retire-serving-id)
      (throw (ex-info "incomplete reconciliation proposal" {:payload payload})))
    {:kind :cloud-murakumo/model-catalog-reconciliation
     :idempotency-key (:idempotency-key proposal)
     :writes [{:op :upsert-serving
               :model-id (:model/id recommended-upsert)
               :body (->catalog-record recommended-upsert endpoint runtime-model-ids)}
              {:op :retire-stale-serving
               :model-id recommended-retire-serving-id
               :body {:id recommended-retire-serving-id
                      :status "registered-not-serving"
                      :superseded-by (:model/id recommended-upsert)}
               :preserve-existing? true}]}))

(defn read-proposal [path]
  (edn/read-string (slurp path)))

(defn- put! [api cacao {:keys [model-id body]}]
  (let [request (-> (HttpRequest/newBuilder (URI/create (str api "/infer/models/" model-id)))
                    (.header "content-type" "application/json")
                    (.header "authorization" (str "CACAO " cacao))
                    (.PUT (HttpRequest$BodyPublishers/ofString (json/generate-string body)))
                    .build)
        response (.send (HttpClient/newHttpClient) request (HttpResponse$BodyHandlers/ofString))]
    (when-not (#{200 201} (.statusCode response))
      (throw (ex-info "catalog write failed"
                      {:model-id model-id :status (.statusCode response) :body (.body response)})))
    {:model-id model-id :status (.statusCode response)}))

(defn- get! [api cacao model-id]
  (let [request (-> (HttpRequest/newBuilder (URI/create (str api "/infer/models/" model-id)))
                    (.header "accept" "application/json")
                    (.header "authorization" (str "CACAO " cacao))
                    .GET .build)
        response (.send (HttpClient/newHttpClient) request (HttpResponse$BodyHandlers/ofString))]
    (case (.statusCode response)
      200 (json/parse-string (.body response) true)
      404 nil
      (throw (ex-info "catalog read failed"
                      {:model-id model-id :status (.statusCode response) :body (.body response)})))))

(defn- retire-write
  [write existing]
  (if (:preserve-existing? write)
    (assoc write :body (merge existing (:body write)))
    write))

(defn- rollback-write
  [upsert prior]
  (assoc upsert :body (or prior
                          (assoc (:body upsert) :status "registered-not-serving"))))

(defn apply!
  "Execute a reviewed plan with a member CACAO. `put-fn` is injectable for
  tests. This function deliberately has no fallback credential and never signs
  on the caller's behalf."
  [plan {:keys [api cacao put-fn get-fn] :or {api default-api put-fn put! get-fn get!}}]
  (when (or (nil? cacao) (empty? cacao))
    (throw (ex-info "MURAKUMO_CACAO is required for --apply" {})))
  (let [[upsert retire] (:writes plan)
        ;; Fetch both snapshots BEFORE changing anything. The stale descriptor is
        ;; merged on retire so status updates cannot discard benchmark/credit
        ;; metadata; the new descriptor is retained for compensation.
        prior-upsert (get-fn api cacao (:model-id upsert))
        prior-retire (get-fn api cacao (:model-id retire))
        retire (retire-write retire prior-retire)]
    (try
      (let [first-write (put-fn api cacao upsert)
            second-write (put-fn api cacao retire)]
        {:idempotency-key (:idempotency-key plan)
         :writes [first-write second-write]})
      (catch Exception cause
        (try
          (put-fn api cacao (rollback-write upsert prior-upsert))
          (catch Exception rollback
            (throw (ex-info "catalog reconciliation and rollback failed"
                            {:cause (ex-message cause) :rollback (ex-message rollback)} rollback))))
        (throw (ex-info "catalog reconciliation failed; upsert was compensated"
                        {:cause (ex-message cause)} cause))))))

(defn -main
  "`bb asi-reconcile [--apply] proposal.edn`. Dry-run is the default."
  [& args]
  (let [[apply? path] (if (= "--apply" (first args)) [true (second args)] [false (first args)])]
    (if-not path
      (binding [*out* *err*] (println "usage: bb asi-reconcile [--apply] <proposal.edn>"))
      (let [plan (reconciliation-plan (read-proposal path))]
        (if apply?
          (pprint/pprint (apply! plan {:cacao (System/getenv "MURAKUMO_CACAO")}))
          (pprint/pprint plan))))))
