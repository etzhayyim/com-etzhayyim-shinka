#!/usr/bin/env bb
;; Shinka Loop C R0 — rank the existing weight family on REAL measured fitness.
;; ADR-2606172200 D6: R0 ranks the family, invents nothing; route ∈
;; {:propose-candidate :insufficient-evidence :excluded}. No fabrication: a candidate
;; missing the t2 task signal (microbench) cannot be scored → :insufficient-evidence.
;; Emits a ranked scorecard (EDN + markdown). Pure data → scorecard (no model run).
(ns rank (:require [clojure.edn :as edn] [clojure.string :as str] [clojure.java.io :as io]))

(def here (-> *file* io/file .getParent))

;; genotypes.edn is now Datomic/Datascript tx-data ([{:db/id -1 :loop-c.genotypes/... ...}])
;; per scripts/edn-datomize.bb (repo-wide EDN datomize effort). Reconstitute the original
;; bare-keyword map here so the rest of this script is unchanged.
(defn- unblob [v]
  (if (string? v)
    (try (let [parsed (edn/read-string v)] (if (coll? parsed) parsed v))
         (catch Exception _ v))
    v))
(defn- reconstitute-entity [tx-data]
  (into {} (map (fn [[k v]] [(keyword (name k)) (unblob v)]))
        (dissoc (first tx-data) :db/id)))

(def G (reconstitute-entity (edn/read-string (slurp (io/file here "genotypes.edn")))))
(def W (:weights G))

(defn flatness
  "map basin-spread → [0,1] flatness term (smaller spread = flatter = higher).
   honest: only when both landscape fields measured."
  [f]
  (let [s (:t2/landscape-basin-spread f)] (when s (/ 1.0 (+ 1.0 (* 10.0 s))))))

(defn throughput-term [f] (when-let [t (:t1/tok-s f)] (min 1.0 (/ t 100.0)))) ; vs a 100 tok/s ref

(defn score
  "Weighted score over ONLY the present terms, renormalised. Requires :t2/microbench
   (the task signal) to be scoreable — else nil (→ insufficient-evidence)."
  [f]
  (when (some? (:t2/microbench f))
    (let [terms (cond-> []
                  (:t2/microbench f)         (conj [:microbench (:t2/microbench f)])
                  (throughput-term f)        (conj [:throughput (throughput-term f)])
                  (flatness f)               (conj [:landscape-flatness (flatness f)])
                  (:t0/feasible f)           (conj [:feasibility 1.0]))
          wsum (reduce + (map (comp W first) terms))]
      (when (pos? wsum)
        (/ (reduce + (map (fn [[k v]] (* (W k) v)) terms)) wsum)))))

(defn evidence-tier [f]
  (cond (some? (:t2/microbench f)) :t2-task-measured
        (some? (:t2/landscape-grad-norm f)) :t2-landscape-only
        (some? (:t3/train-loss f)) :t3-train-only
        :else :t0-feasible-only))

(defn route [s f]
  (cond (not (:t0/feasible f)) :excluded
        (some? s) :propose-candidate          ; scoreable → eligible to propose for fuller eval
        :else :insufficient-evidence))         ; no fabrication

(let [scored (->> (:candidates G)
                  (map (fn [c]
                         (let [f (:fitness c) s (score f)]
                           (assoc c :score s :evidence (evidence-tier f) :route (route s f)))))
                  (sort-by (fn [c] [(if (:score c) 0 1) (- (or (:score c) 0))])))
      ;; Elo: round-robin among scoreable only (R0 — a single comparable cohort)
      elig (filter :score scored)
      out {:schema "shinka.loop-c/scorecard.v0" :ts (:measured-on G)
           :n-candidates (count scored) :n-scoreable (count elig)
           :ranked (mapv #(select-keys % [:id :family :score :evidence :route]) scored)
           :note "R0: ranks existing family on REAL measured fitness only; unmeasured → insufficient-evidence; no architecture invented; promotion to train/deploy is member-CACAO-gated (ADR-2606172200 D5)."}]
  (spit (io/file here "scorecard.edn") (with-out-str (clojure.pprint/pprint out)))
  ;; markdown scorecard (the PR-draft artifact)
  (let [md (StringBuilder.)]
    (.append md "# Shinka Loop C R0 — family ranking (real fitness)\n\n")
    (.append md (str "_" (:ts out) " · " (:n-scoreable out) "/" (:n-candidates out)
                     " scoreable · ADR-2606172200 R0 · ranks the existing family, invents nothing_\n\n"))
    (.append md "| rank | candidate | family | score | evidence | route |\n|---|---|---|---|---|---|\n")
    (doseq [[i c] (map-indexed vector scored)]
      (.append md (format "| %d | %s | %s | %s | %s | %s |\n" (inc i) (:id c) (name (:family c))
                          (if (:score c) (format "%.3f" (:score c)) "—")
                          (name (:evidence c)) (name (:route c)))))
    (.append md "\n**Honesty:** only `maxwell-diffusion-1` has the measured task signal (e7m micro 0.80) → scoreable. ")
    (.append md "`maxwell-1` has a measured loss-landscape + train-loss but no microbench → insufficient until benched. ")
    (.append md "`oka-mmsheaf` / `baien-1.58` have no measured task fitness (oka = R0, no weights). ")
    (.append md "No architecture is invented or promoted; promotion is member-CACAO-gated.\n")
    (spit (io/file here "scorecard.md") (str md)))
  (println "Loop C R0 scorecard:")
  (doseq [c scored]
    (println (format "  %-22s %-10s score=%s route=%s" (:id c) (name (:family c))
                     (if (:score c) (format "%.3f" (:score c)) "—") (name (:route c)))))
  (println "wrote scorecard.edn + scorecard.md"))
