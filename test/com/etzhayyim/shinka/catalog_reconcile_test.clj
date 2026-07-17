(ns com.etzhayyim.shinka.catalog-reconcile-test
  (:require [clojure.test :refer [deftest is testing]]
            [com.etzhayyim.shinka.catalog-reconcile :as reconcile]))

(def proposal
  {:idempotency-key "shinka/model-reconcile/qwen-agentworld-35b-a3b"
   :payload {:requires-member-cacao true
             :catalog-model-id "qwen-agentworld-35b-a3b"
             :endpoint "https://infer.murakumo.cloud/v1/chat/completions"
             :runtime-model-ids ["/home/gad/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"]
             :recommended-retire-serving-id "qwen-agentworld-35b-a3b"
             :recommended-upsert {:model/id "qwen3.6-35b-a3b"
                                  :model/family "Qwen3.6" :model/format :gguf
                                  :model/layers 40 :model/params-b 35 :model/active-params-b 3
                                  :model/context 262144 :model/weight-bytes 21200000000}}})

(deftest reconciliation-plan-is-explicit-and-ordered
  (let [plan (reconcile/reconciliation-plan proposal)]
    (is (= [:upsert-serving :retire-stale-serving] (mapv :op (:writes plan))))
    (is (= "qwen3.6-35b-a3b" (get-in plan [:writes 0 :model-id])))
    (is (= "serving" (get-in plan [:writes 0 :body :status])))
    (is (= "registered-not-serving" (get-in plan [:writes 1 :body :status])))
    (is (true? (get-in plan [:writes 1 :preserve-existing?])))))

(deftest apply-needs-a-member-cacao-and-uses-injected-writer
  (let [plan (reconcile/reconciliation-plan proposal)
        calls (atom [])]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"CACAO"
                          (reconcile/apply! plan {})))
    (is (= 2 (count (:writes (reconcile/apply! plan
                                                {:cacao "member-cacao"
                                                 :get-fn (fn [_ _ model-id]
                                                           (when (= model-id "qwen-agentworld-35b-a3b")
                                                             {:id model-id :family "Qwen-AgentWorld" :tok-s 61.5}))
                                                 :put-fn (fn [api cacao write]
                                                           (swap! calls conj [api cacao write])
                                                           {:model-id (:model-id write) :status 200})})))))
    (is (= "member-cacao" (second (first @calls))))
    (is (= "Qwen-AgentWorld" (get-in (second @calls) [2 :body :family])))))

(deftest failed-retirement-compensates-the-new-serving-entry
  (let [plan (reconcile/reconciliation-plan proposal)
        writes (atom [])]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"compensated"
                          (reconcile/apply! plan
                                            {:cacao "member-cacao"
                                             :get-fn (constantly nil)
                                             :put-fn (fn [_ _ write]
                                                       (swap! writes conj write)
                                                       (when (= :retire-stale-serving (:op write))
                                                         (throw (ex-info "retire failed" {})))
                                                       {:model-id (:model-id write) :status 200})})))
    (is (= "registered-not-serving" (get-in @writes [2 :body :status])))))
