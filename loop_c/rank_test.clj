#!/usr/bin/env bb
;; Loop C R0 invariants (no-fabrication + weight-sum + routing). Run: bb rank_test.clj
(ns rank-test (:require [clojure.edn :as edn] [clojure.java.io :as io]))
(def here (-> *file* io/file .getParent))
(def G (edn/read-string (slurp (io/file here "genotypes.edn"))))
(def errs (atom 0))
(defn check [name ok] (if ok (println "  ok " name) (do (println "  FAIL" name) (swap! errs inc))))

;; run the ranker, then inspect its artifact
(load-file (str (io/file here "rank.clj")))
(def sc (edn/read-string (slurp (io/file here "scorecard.edn"))))
(def by-id (into {} (map (juxt :id identity) (:ranked sc))))

(check "Σ weights = 1.0" (< (abs (- 1.0 (reduce + (vals (:weights G))))) 1e-9))
(check "schema present" (= "shinka.loop-c/scorecard.v0" (:schema sc)))
;; no-fabrication: every candidate WITHOUT a measured microbench must be insufficient-evidence
(let [no-bench (->> (:candidates G) (remove #(some? (:t2/microbench (:fitness %)))) (map :id))]
  (check "no-fabrication: unmeasured → insufficient-evidence"
         (every? #(= :insufficient-evidence (:route (by-id %))) no-bench)))
;; the only measured-task candidate is scoreable + proposed
(check "measured candidate is scoreable"
       (some? (:score (by-id "maxwell-diffusion-1"))))
(check "measured candidate routes :propose-candidate"
       (= :propose-candidate (:route (by-id "maxwell-diffusion-1"))))
;; ranking: scoreable outranks unscoreable (appears first)
(check "scoreable ranked above insufficient"
       (< (.indexOf (mapv :id (:ranked sc)) "maxwell-diffusion-1")
          (.indexOf (mapv :id (:ranked sc)) "maxwell-1")))
;; oka (R0, no weights) must never be scored
(check "oka (R0) never scored" (nil? (:score (by-id "oka-mmsheaf"))))

(println (if (zero? @errs) "ALL PASS" (str @errs " FAIL")))
(System/exit (if (zero? @errs) 0 1))
