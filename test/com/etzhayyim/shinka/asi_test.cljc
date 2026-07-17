(ns com.etzhayyim.shinka.asi-test
  (:require [clojure.test :refer [deftest is]]
            [com.etzhayyim.shinka.asi :as asi]
            [kotoba.fleet.evolution :as fleet-evolution]))

(def GiB (* 1024 1024 1024))
(def attested (zipmap fleet-evolution/required-attestations (repeat true)))
(def model {:model/id "shinka-eval" :model/layers 2 :model/weight-bytes (* 2 GiB)
            :model/kv-heads 8})
(def node {:name "reuben" :health :healthy :caps #{:inference}
           :mem-bytes (* 16 GiB) :link-gbps 10})
(def evidence-cid "bafybeigdyrzt5q6h3l6s4v6g3y5l6e5xxqg3o2h4k5c2z7p3q6b3j6q5ae")

(deftest asi-beat-dispatches-but-never-auto-merges
  (let [plan (asi/beat-plan {:candidate-id "c-1" :benchmark-id "b-1" :prompts ["check"]
                             :model model :nodes [node] :now "2026-07-10T03:00:00Z"
                             :attestations attested :evidence-cid evidence-cid
                             :baseline-score 0.50 :candidate-score 0.57 :benchmark-steps 300})]
    (is (= :ready (get-in plan [:dispatch :status])))
    (is (= :human-signoff (get-in plan [:promotion :status])))
    (is (= [:murakumo/enqueue :kotoba-fleet/submit-proposal :mst/put-record]
           (mapv :op (:effects plan))))
    (is (= "asi-c-1-b-1" (get-in plan [:record :rkey])))
    (is (= "shinka/c-1/b-1/0" (get-in plan [:effects 0 :payload :jobs 0 :idempotency-key])))
    (is (false? (get-in plan [:record :autoMerge])))))

(deftest asi-beat-holds-when-the-fleet-is-down
  (let [plan (asi/beat-plan {:candidate-id "c-2" :benchmark-id "b-2" :prompts ["check"]
                             :model model :nodes [(assoc node :health :down)]
                             :attestations attested :evidence-cid evidence-cid
                             :baseline-score 0.50 :candidate-score 0.51 :benchmark-steps 10})]
    (is (= :blocked (get-in plan [:dispatch :status])))
    (is (= [:mst/put-record] (mapv :op (:effects plan))))))

(deftest asi-beat-refuses-incomplete-or-unbounded-input
  (let [plan (asi/beat-plan {:candidate-id "" :benchmark-id "b-3"
                             :prompts (vec (repeat 33 "too many"))
                             :model {:model/id "bad" :model/layers 0 :model/weight-bytes 0}
                             :nodes [node]})]
    (is (= :blocked (get-in plan [:dispatch :status])))
    (is (= :invalid-input (get-in plan [:dispatch :reason])))
    (is (= #{:missing-candidate-id :too-many-prompts :invalid-model}
           (set (get-in plan [:dispatch :errors]))))
    (is (empty? (:effects plan)))))

(deftest asi-beat-refuses-a-live-catalog-model-that-is-not-serving
  (let [plan (asi/beat-plan {:candidate-id "c-4" :benchmark-id "b-4" :prompts ["check"]
                             :model model :nodes [node]
                             :model-availability {:status :not-serving
                                                  :available-models ["actual-serving-model"]}})]
    (is (= :blocked (get-in plan [:dispatch :status])))
    (is (= :model-not-serving (get-in plan [:dispatch :reason])))
    (is (= ["actual-serving-model"] (get-in plan [:dispatch :available-models])))
    (is (empty? (:effects plan)))))

(deftest asi-beat-uses-a-catalog-serving-gateway-without-fleet-shard-assumptions
  (let [plan (asi/beat-plan {:candidate-id "c-5" :benchmark-id "b-5" :prompts ["check"]
                             :model model :nodes []
                             :model-availability {:status :serving :execution-mode :gateway
                                                  :endpoint "https://infer.example/v1/chat/completions"}})]
    (is (= :ready (get-in plan [:dispatch :status])))
    (is (= :gateway (get-in plan [:dispatch :execution])))
    (is (= [:cloud-murakumo/chat-completions :mst/put-record]
           (mapv :op (:effects plan))))
    (is (= {:model "shinka-eval"
            :messages [{:role "user" :content "check"}]
            :max_tokens 512
            :metadata {:candidate-id "c-5" :benchmark-id "b-5"
                       :prompt-id "b-5-0" :reproducible true}}
           (get-in plan [:effects 0 :payload :jobs 0 :request])))))

(deftest asi-beat-refuses-a-serving-catalog-entry-without-runtime-model-proof
  (let [plan (asi/beat-plan {:candidate-id "c-6" :benchmark-id "b-6" :prompts ["check"]
                             :model model :nodes []
                             :model-availability {:status :runtime-unverified
                                                  :runtime-model-ids ["other-model"]}})]
    (is (= :blocked (get-in plan [:dispatch :status])))
    (is (= :runtime-model-unverified (get-in plan [:dispatch :reason])))
    (is (= ["other-model"] (get-in plan [:dispatch :runtime-model-ids])))
    (is (= [:kotoba-fleet/submit-proposal] (mapv :op (:effects plan))))
    (is (= "shinka/model-reconcile/shinka-eval"
           (get-in plan [:effects 0 :payload :idempotency-key])))
    (is (true? (get-in plan [:effects 0 :payload :payload :requires-member-cacao])))))

(deftest asi-beat-attaches-the-reviewed-qwen3-dot-6-reconciliation-payload
  (let [runtime-id "/home/gad/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
        plan (asi/beat-plan {:candidate-id "c-7" :benchmark-id "b-7" :prompts ["check"]
                             :model {:model/id "qwen-agentworld-35b-a3b" :model/layers 40
                                     :model/weight-bytes 22100000000}
                             :nodes []
                             :model-availability {:status :runtime-unverified
                                                  :runtime-model-ids [runtime-id]}})
        payload (get-in plan [:effects 0 :payload :payload])]
    (is (= :replace-serving-model (:recommended-action payload)))
    (is (= "qwen3.6-35b-a3b" (get-in payload [:recommended-upsert :model/id])))
    (is (= 21200000000 (get-in payload [:recommended-upsert :model/weight-bytes])))
    (is (= "qwen-agentworld-35b-a3b" (:recommended-retire-serving-id payload)))))
