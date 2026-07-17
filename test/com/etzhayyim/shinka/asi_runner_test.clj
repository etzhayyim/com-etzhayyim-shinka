(ns com.etzhayyim.shinka.asi-runner-test
  (:require [clojure.test :refer [deftest is]]
            [com.etzhayyim.shinka.asi-runner :as runner]
            [kotoba.fleet.evolution :as fleet-evolution]))

(def GiB (* 1024 1024 1024))
(def attested (zipmap fleet-evolution/required-attestations (repeat true)))
(def model {:model/id "shinka-eval" :model/layers 2 :model/weight-bytes (* 2 GiB)
            :model/kv-heads 8})
(def node {:name "reuben" :health :healthy :caps #{:inference}
           :mem-bytes (* 16 GiB) :link-gbps 10})

(deftest preflight-reports-a-fleet-blocker
  (let [result (runner/preflight {:candidate-id "c-1" :benchmark-id "b-1"
                                  :prompts ["check"] :model model
                                  :nodes [(assoc node :health :down)]
                                  :attestations attested})]
    (is (false? (:ready? result)))
    (is (= :no-healthy-inference-nodes (:reason result)))
    (is (= :blocked (get-in result [:plan :dispatch :status])))))

(deftest preflight-marks-a-healthy-bounded-run-ready
  (let [result (runner/preflight {:candidate-id "c-2" :benchmark-id "b-2"
                                  :prompts ["check"] :model model :nodes [node]
                                  :attestations attested})]
    (is (:ready? result))
    (is (= :fleet-ready (:reason result)))
    (is (= :ready (get-in result [:plan :dispatch :status])))))

(deftest live-preflight-normalizes-cloud-observations-and-fails-closed
  (let [input {:candidate-id "c-3" :benchmark-id "b-3" :prompts ["check"]
               :model model :attestations attested}
        healthy (runner/live-preflight input
                                       (constantly {:fleet {:per-node-shard-ceiling-gb 10
                                                            :nodes [{:id "mnode-1" :status "up"
                                                                     :roles ["llama"] :ram-gb 16}]}
                                                    :models [{:id "shinka-eval" :status "serving"
                                                              :endpoint "https://infer.example/v1/chat/completions"}]
                                                    :runtime-models {"https://infer.example/v1/chat/completions"
                                                                     ["shinka-eval"]}}))
        unavailable (runner/live-preflight input
                                            (constantly {:fleet {:nodes []}
                                                         :models [{:id "other" :status "serving"}]
                                                         :runtime-models {}}))
        mismatched (runner/live-preflight input
                                           (constantly {:fleet {:nodes []}
                                                        :models [{:id "shinka-eval" :status "serving"
                                                                  :endpoint "https://infer.example/v1/chat/completions"}]
                                                        :runtime-models {"https://infer.example/v1/chat/completions"
                                                                         ["different-model"]}}))
        broken (runner/live-preflight input (fn [] (throw (ex-info "offline" {}))))]
    (is (:ready? healthy))
    (is (= :ready (get-in healthy [:plan :dispatch :status])))
    (is (= :gateway (get-in healthy [:plan :dispatch :execution])))
    (is (= {:observed-nodes 1 :healthy-inference-nodes 1 :healthy-names ["mnode-1"]}
           (:fleet healthy)))
    (is (= :model-not-serving (:reason unavailable)))
    (is (= :runtime-model-unverified (:reason mismatched)))
    (is (false? (:ready? broken)))
    (is (= :fleet-observation-failed (:reason broken)))))
