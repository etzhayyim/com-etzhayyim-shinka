(ns com.etzhayyim.shinka.asi
  "Shinka's bounded ASI outer loop.

  Capability generation is dispatched only through cloud-murakumo.  Promotion
  is governed by kotoba-fleet and is therefore proposal/signoff-only; this actor
  never writes a git ref, model registry, or weight directly."
  (:require [clojure.string :as str]
            [kotoba.fleet.evolution :as fleet-evolution]
            [murakumo.infer.evolution :as murakumo-evolution]
            [com.etzhayyim.shinka.murakumo :as actor]))

(def self-evolution-collection "com.etzhayyim.apps.standard.shinkaEvolution")
(def max-prompts 32)
(def max-tokens-per-prompt 4096)

(def runtime-model-recommendations
  "Observed runtime model IDs → reviewed canonical Murakumo registry entries.
  This table is intentionally small and exact-match-only. A novel runtime name
  remains human review work; Shinka must never guess a semantic model alias."
  {"/home/gad/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
   {:model/id "qwen3.6-35b-a3b"
    :model/family "Qwen3.6"
    :model/format :gguf
    :model/layers 40
    :model/params-b 35
    :model/active-params-b 3
    :model/context 262144
    :model/weight-bytes 21200000000}})

(declare beat-rkey)

(defn input-errors
  "Validate the bounded ASI-beat contract before any dispatch plan is created.
  This is deliberately stricter than the underlying generic fleet planner:
  Shinka needs stable identities and finite evaluation budgets for auditability."
  [{:keys [candidate-id benchmark-id prompts model max-tokens]}]
  (cond-> []
    (str/blank? (str candidate-id)) (conj :missing-candidate-id)
    (str/blank? (str benchmark-id)) (conj :missing-benchmark-id)
    (or (not (sequential? prompts)) (empty? prompts)) (conj :missing-prompts)
    (and (sequential? prompts) (> (count prompts) max-prompts)) (conj :too-many-prompts)
    (and (sequential? prompts) (some #(str/blank? (str %)) prompts)) (conj :blank-prompt)
    (not (and (map? model) (string? (:model/id model))
              (pos? (or (:model/layers model) 0))
              (pos? (or (:model/weight-bytes model) 0)))) (conj :invalid-model)
    (and max-tokens (or (not (integer? max-tokens))
                        (not (<= 1 max-tokens max-tokens-per-prompt)))) (conj :invalid-max-tokens)))

(defn- blocked-plan
  [candidate-id benchmark-id now reason & [details]]
  {:cell :shinka-asi-beat
   :dispatch (merge {:status :blocked :reason reason :jobs []} details)
   :promotion {:status :blocked :reason reason}
   :record {:$type self-evolution-collection
            :rkey (beat-rkey candidate-id benchmark-id)
            :candidateId candidate-id :benchmarkId benchmark-id :computedAt now
            :appendOnly true :autoMerge false}
   :effects []})

(defn beat-rkey
  "Stable identity for one candidate/benchmark pair. Replaying the same beat
  targets the same append-only record key and carries the same relay keys."
  [candidate-id benchmark-id]
  (str "asi-" (actor/safe-rkey candidate-id) "-" (actor/safe-rkey benchmark-id)))

(defn- effect [op payload]
  {:op op :actor actor/actor-did :payload payload})

(defn- reconciliation-proposal-effect
  "Ask the independent fleet governor for a human-signed catalog correction.
  This effect cannot alter the cloud catalog; its sole outcome is an auditable
  proposal that a member may inspect, sign, materialize, or reject."
  [candidate-id model now model-availability]
  (let [model-id (:model/id model)
        key (str "shinka/model-reconcile/" model-id)
        runtime-ids (:runtime-model-ids model-availability)
        recommended (some runtime-model-recommendations runtime-ids)]
    (effect :kotoba-fleet/submit-proposal
            {:work (str "cloud-murakumo/model-catalog/" model-id)
             :agent actor/actor-did
             :idempotency-key key
             :payload {:kind :cloud-murakumo/model-catalog-reconciliation
                       :requires-member-cacao true
                       :candidate-id candidate-id
                       :catalog-model-id model-id
                       :endpoint (:endpoint model-availability)
                       :runtime-model-ids runtime-ids
                       :recommended-action (if recommended :replace-serving-model :manual-review)
                       :recommended-upsert recommended
                       :recommended-retire-serving-id (when recommended model-id)
                       :observed-at now}})))

(defn- gateway-dispatch-plan
  "Build OpenAI-compatible requests for a catalog-confirmed serving gateway.
  This is still an effect plan, never an HTTP call.  The stable idempotency key
  is retained so a host executor can safely retry a timed-out request."
  [{:keys [candidate-id benchmark-id prompts max-tokens model model-availability]}]
  {:status :ready
   :execution :gateway
   :endpoint (:endpoint model-availability)
   :jobs (mapv (fn [prompt index]
                 {:idempotency-key (str "shinka/" candidate-id "/" benchmark-id "/" index)
                  :request {:model (:model/id model)
                            :messages [{:role "user" :content prompt}]
                            :max_tokens (or max-tokens 512)
                            :metadata {:candidate-id candidate-id
                                       :benchmark-id benchmark-id
                                       :prompt-id (str benchmark-id "-" index)
                                       :reproducible true}}})
               prompts (range))})

(defn beat-plan
  "Plan one auditable Shinka ASI beat. No network or state mutation occurs here.

  Input contains a live fleet observation, model descriptor, bounded benchmark
  prompts, and (when results have arrived) candidate evidence.  The caller
  executes `:murakumo/enqueue` through cloud-murakumo and appends the records.
  A passing candidate becomes a human-signoff proposal, never an auto-merge."
  [{:keys [candidate-id benchmark-id prompts model nodes now model-availability] :as input}]
  (let [errors (input-errors input)
        rkey (beat-rkey candidate-id benchmark-id)]
    (cond
      (seq errors)
      (blocked-plan candidate-id benchmark-id now :invalid-input {:errors errors})

      (= :not-serving (:status model-availability))
      (blocked-plan candidate-id benchmark-id now :model-not-serving
                    {:model-id (:model/id model)
                     :available-models (:available-models model-availability)})

      (= :runtime-unverified (:status model-availability))
      (assoc (blocked-plan candidate-id benchmark-id now :runtime-model-unverified
                           {:model-id (:model/id model)
                            :runtime-model-ids (:runtime-model-ids model-availability)})
             :effects [(reconciliation-proposal-effect candidate-id model now model-availability)])

      :else
      (let [dispatch (if (= :gateway (:execution-mode model-availability))
                       (gateway-dispatch-plan input)
                       (murakumo-evolution/dispatch-plan
                        {:candidate-id candidate-id :benchmark-id benchmark-id
                         :prompts prompts :model model :nodes nodes
                         :max-tokens (:max-tokens input)}))
        promotion (fleet-evolution/promotion-verdict input)
        base-record {:$type self-evolution-collection
                     :rkey rkey
                     :candidateId candidate-id :benchmarkId benchmark-id
                     :computedAt now :dispatchStatus (:status dispatch)
                     :promotionStatus (:status promotion)
                     :appendOnly true :autoMerge false}
        dispatch-effects (when (= :ready (:status dispatch))
                           [(if (= :gateway (:execution dispatch))
                              (effect :cloud-murakumo/chat-completions
                                      {:endpoint (:endpoint dispatch) :jobs (:jobs dispatch)})
                              (effect :murakumo/enqueue {:jobs (:jobs dispatch)
                                                         :strategy (:strategy dispatch)}))])
        promotion-effects (when (= :human-signoff (:status promotion))
                            [(effect :kotoba-fleet/submit-proposal
                                     {:work (str "shinka/" candidate-id)
                                      :agent actor/actor-did
                                      :payload (assoc promotion :candidate-id candidate-id)})])]
    {:cell :shinka-asi-beat
     :dispatch dispatch
     :promotion promotion
     :record base-record
     :effects (vec (concat dispatch-effects promotion-effects
                           [(effect :mst/put-record {:collection self-evolution-collection
                                                     :rkey rkey :record base-record})]))}))))
