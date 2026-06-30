(ns com.etzhayyim.shinka.murakumo-test
  (:require [clojure.test :refer [deftest is testing]]
            [com.etzhayyim.shinka.murakumo :as shinka]))

(deftest mood-classification
  (is (= :stressed (shinka/classify-mood {:joy 90 :stress 70})))
  (is (= :joyful (shinka/classify-mood {:joy 60 :stress 20})))
  (is (= :calm (shinka/classify-mood {:calm 60})))
  (is (= :grateful (shinka/classify-mood {:gratitude 60})))
  (is (= :focused (shinka/classify-mood {:focus 60})))
  (is (= :neutral (shinka/classify-mood {}))))

(deftest cadence-resolution
  (testing "joyful cadence"
    (is (= {:mood :joyful
            :should-post true
            :should-engage true
            :should-drill false
            :should-validate false
            :should-analyze false}
           (shinka/cadence-flags {:axes {:joy 80} :elapsed-minutes 30}))))
  (testing "stressed cadence suppresses post and engage"
    (is (= {:mood :stressed
            :should-post false
            :should-engage false
            :should-drill true
            :should-validate false
            :should-analyze false}
           (shinka/cadence-flags {:axes {:stress 80} :elapsed-minutes 30})))))

(deftest heartbeat-record-shape
  (let [out (shinka/heartbeat-record {:adherent-did "did:web:alice.example"
                                      :axes {:joy 61}
                                      :elapsed-minutes nil
                                      :now "2026-06-29T00:00:00Z"})]
    (is (= shinka/heartbeat-collection (:collection out)))
    (is (= "heartbeat-alice.example" (:rkey out)))
    (is (= "did:web:alice.example" (get-in out [:record :actorDid])))
    (is (= :joyful (get-in out [:record :mood])))))

(deftest legacy-cell-effect-plans
  (let [input {:adherent-did "did:web:alice.example"
               :axes {:focus 80}
               :elapsed-minutes 180
               :now "2026-06-29T00:00:00Z"
               :proposed-evolution {:claim-id "claim-1" :proposed-level 2}}
        plan (shinka/tick-plan input)]
    (is (= :shinka-tick (:cell plan)))
    (is (= 4 (count (:effects plan))))
    (is (= [shinka/observe-collection
            shinka/heartbeat-collection
            shinka/validate-collection
            shinka/evolution-collection]
           (mapv :collection (:effects plan))))
    (is (= :focused (get-in plan [:steps 0 :adherent :cadence :mood])))
    (is (= "evolution-claim-1" (-> plan :effects last :rkey)))))
