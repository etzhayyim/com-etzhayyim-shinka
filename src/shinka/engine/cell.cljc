(ns shinka.cells.shinka-engine.cell
  "ShinkaEvolutionCell — Shinka capability-evolution cell (Loop A, S0).
  1:1 Clojure port of `src/shinka/engine/cell.py` (ADR-2606142200).

  A single super-step graph that maps the DeepMind co-scientist
  generate→debate→evolve cycle onto etzhayyim primitives:

      propose (Generation) → reflect (Reflection/critic, Charter G1-G8 pre-scan)
        → cluster (Proximity, diversity) → rank (Ranking, Elo pairwise debate)
        → recombine (Evolution) → synthesize (Meta-review, PR draft) → emit (datoms)

  The cell is deterministic and LLM-free at S0 so it runs and tests offline; the
  Murakumo debate is a typed hook (`murakumo-debate`) that fails OPEN to the
  deterministic kernel (Murakumo-only invariant I3 — never a commercial call).

  Invariants (mirrored from the Python source):
    I1  every fact emitted is an append-only `:db/add` datom — `datom` REFUSES
        to build a `:db/retract`. The evolution history cannot be rewritten.
    I2  `node-synthesize` emits a PR draft with member-signed=false / auto-merge=false.
        `is-committable` is false until a member CACAO capability is attached.
    I3  inference resolves Murakumo-only; the offline kernel is the fail-open path.

  House style (per kamado feedstock_guard.cljc): Python ':…' keyword strings stay
  strings (datom attributes/ops, `:db/add`); pure fns thread state→state; the
  Python ValueError edge → ex-info; portable .cljc, loads under babashka. There is
  no LangGraph in cljc — the Python cell already falls back to an identical
  sequential super-step driver when StateGraph is None, so `solve` runs that."
  (:require [clojure.string :as str]))

;; --------------------------------------------------------------------------- ;;
;; Charter Rider G1-G8 scanner (local fallback only)
;;
;; The Python source best-effort imports the canonical etzhayyim-organism scanner
;; and falls open to a minimal local scan. The canonical scanner is a Python module
;; not available from cljc, so this port carries ONLY the conservative local subset
;; (ADR-2605192200) — same guarantee: never pass an obviously-prohibited proposal.
;; --------------------------------------------------------------------------- ;;

(def ^:private PROHIBITED-SIGNALS
  ["runpod" "vertex ai" "aws bedrock" "commercial gpu" "weapon design"
   "covert force" "child sexual" "transfer() land" "setowner"])

(defn local-scan-ok
  "Python _local_scan_ok: true iff no prohibited signal occurs in (lower-cased) text."
  [text]
  (let [low (str/lower-case (str text))]
    (not (some #(str/includes? low %) PROHIBITED-SIGNALS))))

;; In a bare cljc checkout the canonical scanner is unavailable → fail open to local.
(defn scan-ok [text] (local-scan-ok text))

;; --------------------------------------------------------------------------- ;;
;; Data model
;; --------------------------------------------------------------------------- ;;

(def ^:private DEFAULT-ELO 1200.0)
(def ^:private ELO-K 32.0)

;; The four candidate-mutation kinds, in propose-cycle order (Python `kinds`).
(def ^:private KINDS ["cell-impl" "schema-upgrade" "corpus-pair" "code-fix"])

(defn make-proposal
  "A candidate mutation (Python dataclass `Proposal`) as an immutable map.
  Defaults mirror the dataclass field defaults."
  [{:keys [pid kind body rationale source-refs]}]
  {:pid pid
   :kind kind
   :body body
   :rationale rationale
   :source-refs (vec (or source-refs []))   ;; Datom-log retrieval anchors
   :charter-ok nil
   :review-score 0.0                          ;; Reflection correctness heuristic, 0..1
   :elo DEFAULT-ELO
   :cluster-id nil
   :is-duplicate false})

(defn proposal-text
  "Python Proposal.text(): kind\\nbody\\nrationale."
  [p]
  (str (:kind p) "\n" (:body p) "\n" (:rationale p)))

(defn make-evolution-state
  "State threaded through the Shinka evolution super-step graph (Python `EvolutionState`)."
  [{:keys [task context-refs n-propose member-cacao]}]
  {:task task
   :context-refs (vec (or context-refs []))
   :n-propose (or n-propose 4)
   :proposals []
   :rejected []                ;; charter-failed (kept as evidence)
   :debates []
   :merged nil
   :meta-review ""
   :pr-draft nil
   :corpus-candidates []       ;; Loop-B feed (dry-run)
   :datoms []
   :member-cacao member-cacao  ;; opaque CACAO capability; nil ⇒ not committable
   :error-msg nil})

;; --------------------------------------------------------------------------- ;;
;; Helpers
;; --------------------------------------------------------------------------- ;;

(defn datom
  "Build an append-only datom. I1: refuses anything but :db/add (the Python
  ValueError edge → ex-info)."
  ([e a v] (datom e a v ":db/add"))
  ([e a v op]
   (when-not (= op ":db/add")
     (throw (ex-info
             (str "shinka evolution history is append-only (I1): refused op "
                  (pr-str op) "; proposals/verdicts are facts, never retractions "
                  "(ADR-2606142200)")
             {:op op})))
   {:e e :a a :v v :op op}))

(defn elo-update
  "Standard Elo update for a pairwise debate (AlphaGo-style ranking, co-scientist).
  Returns [ra' rb'] (the Python tuple)."
  ([ra rb a-won] (elo-update ra rb a-won ELO-K))
  ([ra rb a-won k]
   (let [ea (/ 1.0 (+ 1.0 (Math/pow 10.0 (/ (- rb ra) 400.0))))
         eb (- 1.0 ea)
         sa (if a-won 1.0 0.0)
         sb (- 1.0 sa)]
     [(+ ra (* k (- sa ea)))
      (+ rb (* k (- sb eb)))])))

(defn stable-score
  "Deterministic 0..1 quality proxy (LLM-free kernel stand-in for a judge).
  A stable hash over the text — `h = (h*131 + ord(ch)) & 0xFFFFFFFF`, then
  `(h % 1000) / 1000.0`. Replayable in tests; byte-identical to the Python kernel."
  [text]
  (let [h (reduce (fn [h ch]
                    (bit-and (+ (* h 131) (long (int ch))) 0xFFFFFFFF))
                  0
                  (str text))]
    (/ (double (mod h 1000)) 1000.0)))

;; --- small formatting helpers to match Python f-string output --------------- ;;

(defn- round-to
  "Round to n decimal places. NOTE: Python round() is banker's rounding;
  this is round-half-up. Hash-derived scores rarely tie at the boundary, so the
  emitted datom values match in practice."
  ;; TODO(port): exact Python banker's rounding (round-half-to-even) if a datom
  ;; value ever lands exactly on a .5 boundary at the 3rd/1st decimal.
  [x n]
  (let [f (Math/pow 10.0 n)]
    (/ (double (Math/round (* (double x) f))) f)))

(defn- fmt0
  "Python f\"{x:.0f}\" — fixed-point, zero decimals."
  [x]
  (format "%.0f" (double x)))

(defn- py-repr-str
  "Python repr() of a string for the meta-review (single-quoted)."
  ;; TODO(port): escape embedded quotes/specials exactly as CPython repr if a
  ;; task string ever contains a single quote.
  [s]
  (str "'" s "'"))

;; --------------------------------------------------------------------------- ;;
;; Node functions (Co-scientist agents) — pure: state-in, state-out
;; --------------------------------------------------------------------------- ;;

(defn node-propose
  "Generation: emit n candidate mutations grounded in retrieved context.

  When a `sampler` is supplied (Research Track A FleetSampler), each proposal BODY
  is drawn via fleet best-of-N; sampler=nil keeps the offline kernel. Murakumo-only
  (I3) — the sampler resolves to fleet endpoints.

  Port note: the Python sampler is an object with `.best_of_n(prompt, n=3)`. Here a
  non-nil `sampler` is treated as a fn `(sampler kind i task context-refs) => [body
  rationale]`; nil drives the deterministic kernel exactly as the Python source."
  ([state] (node-propose state nil))
  ([state sampler]
   (let [task (:task state)
         context-refs (:context-refs state)
         n (:n-propose state)]
     (reduce
      (fn [st i]
        (let [kind (nth KINDS (mod i (count KINDS)))
              pid (str "p" i)
              [body rationale]
              (if (some? sampler)
                ;; TODO(port): faithful FleetSampler best-of-N object shape; the
                ;; kernel (sampler=nil) path is the S0 deterministic contract.
                (sampler kind i task context-refs)
                [(str "[" kind "] candidate " i " for task: " task)
                 (str "grounded in " (count context-refs) " Datom-log refs; angle " i)])
              p (make-proposal {:pid pid :kind kind :body body
                                :rationale rationale :source-refs context-refs})]
          (-> st
              (update :proposals conj p)
              (update :datoms conj (datom (str "shinka:proposal/" pid) ":proposal/kind" kind))
              (update :datoms conj (datom (str "shinka:proposal/" pid) ":proposal/task" task)))))
      state
      (range n)))))

(defn node-reflect
  "Reflection (virtual peer review): Charter G1-G8 pre-scan + correctness score.
  Charter-failing proposals move to `rejected` but are STILL recorded as datoms
  (I1) — a rejection is evidence, not a deletion."
  [state]
  (reduce
   (fn [st p]
     (let [text (proposal-text p)
           charter-ok (scan-ok text)
           review-score (stable-score text)
           p' (assoc p :charter-ok charter-ok :review-score review-score)
           st (-> st
                  (update :datoms conj
                          (datom (str "shinka:proposal/" (:pid p')) ":proposal/charter-ok" charter-ok))
                  (update :datoms conj
                          (datom (str "shinka:proposal/" (:pid p')) ":proposal/review-score"
                                 (round-to review-score 3))))]
       (if charter-ok
         (update st :proposals conj p')
         (-> st
             (update :rejected conj p')
             (update :datoms conj
                     (datom (str "shinka:proposal/" (:pid p')) ":proposal/status" "charter-rejected"))))))
   ;; survivors are rebuilt from scratch (Python rebinds state.proposals = survivors)
   (assoc state :proposals [])
   (:proposals state)))

(defn node-cluster
  "Proximity: cluster by kind for diversity; flag duplicates (keep best per cluster)."
  [state]
  (let [props (:proposals state)
        by-kind (group-by :kind props)
        kinds-sorted (sort (keys by-kind))
        ;; iterate sorted (kind, group) pairs → cluster id = index; within a group
        ;; sort by review-score descending and flag rank>0 as duplicate.
        {:keys [assign datoms]}
        (reduce
         (fn [acc [cid kind]]
           (let [group (sort-by (comp - :review-score) (get by-kind kind))]
             (reduce
              (fn [a [rank-in-group p]]
                (-> a
                    (assoc-in [:assign (:pid p)]
                              {:cluster-id cid :is-duplicate (pos? rank-in-group)})
                    (update :datoms conj
                            (datom (str "shinka:proposal/" (:pid p)) ":proposal/cluster" cid))))
              acc
              (map-indexed vector group))))
         {:assign {} :datoms []}
         (map-indexed vector kinds-sorted))]
    (-> state
        ;; apply attributes back, preserving original proposal order (Python mutates in place)
        (assoc :proposals (mapv #(merge % (get assign (:pid %))) props))
        (update :datoms into datoms))))

(defn murakumo-debate
  "Pairwise scientific debate. I3: Murakumo-only; fails OPEN to the kernel.
  Returns true iff `a` wins. With a live Murakumo `infer` hook (a fn string→string)
  this runs a structured debate; offline it falls back to the deterministic proxy."
  [a b infer]
  (let [verdict
        (when (some? infer)
          (try
            (let [v (-> (infer
                         (str "Adversarially debate which proposal better advances the task "
                              "under the Charter. A:\n" (proposal-text a) "\n\nB:\n"
                              (proposal-text b) "\n\nAnswer exactly 'A' or 'B'."))
                        str str/trim str/upper-case)]
              (cond (str/starts-with? v "A") true
                    (str/starts-with? v "B") false
                    :else nil))
            (catch #?(:clj Exception :cljs :default) _ nil)))]  ;; fail open to kernel
    (if (some? verdict)
      verdict
      ;; Deterministic kernel: higher review score wins; tie broken by stable hash.
      (if (not= (:review-score a) (:review-score b))
        (> (:review-score a) (:review-score b))
        (>= (stable-score (:pid a)) (stable-score (:pid b)))))))

(defn node-rank
  "Ranking: round-robin Elo tournament over the non-duplicate survivors."
  ([state] (node-rank state nil))
  ([state infer]
   (let [contenders (filterv (complement :is-duplicate) (:proposals state))
         n (count contenders)
         pairs (for [i (range n) j (range (inc i) n)] [i j])
         init {:elos (into {} (map (fn [p] [(:pid p) (:elo p)]) contenders))
               :debates [] :datoms []}
         {:keys [elos debates datoms]}
         (reduce
          (fn [acc [i j]]
            (let [a (nth contenders i)
                  b (nth contenders j)
                  ra (get-in acc [:elos (:pid a)])
                  rb (get-in acc [:elos (:pid b)])
                  a-won (murakumo-debate a b infer)
                  [ra' rb'] (elo-update ra rb a-won)
                  winner (if a-won (:pid a) (:pid b))]
              (-> acc
                  (assoc-in [:elos (:pid a)] ra')
                  (assoc-in [:elos (:pid b)] rb')
                  (update :debates conj {:a (:pid a) :b (:pid b) :winner winner})
                  (update :datoms conj
                          (datom (str "shinka:debate/" (:pid a) "-vs-" (:pid b))
                                 ":debate/winner" winner)))))
          init
          pairs)
         ;; per-contender final-elo datom (round 1), in contender order
         elo-datoms (mapv (fn [p]
                            (datom (str "shinka:proposal/" (:pid p)) ":proposal/elo"
                                   (round-to (get elos (:pid p)) 1)))
                          contenders)]
     (-> state
         ;; write back evolved elo onto the proposals (preserving order)
         (assoc :proposals (mapv (fn [p] (if (contains? elos (:pid p))
                                           (assoc p :elo (get elos (:pid p)))
                                           p))
                                 (:proposals state)))
         (update :debates into debates)
         (update :datoms into datoms)
         (update :datoms into elo-datoms)))))

(defn node-recombine
  "Evolution: merge the top-2 Elo contenders into one stronger candidate."
  [state]
  (let [contenders (sort-by (comp - :elo)
                            (filterv (complement :is-duplicate) (:proposals state)))]
    (if (empty? contenders)
      state
      (let [top (first contenders)
            merged
            (if (>= (count contenders) 2)
              (let [second-p (second contenders)]
                (make-proposal
                 {:pid "merged"
                  :kind (:kind top)
                  :body (str (:body top) "\n+ grafted from " (:pid second-p) ": " (:body second-p))
                  :rationale (str "recombination of top-Elo " (:pid top) "(" (fmt0 (:elo top)) ") "
                                  "+ " (:pid second-p) "(" (fmt0 (:elo second-p)) ")")
                  :source-refs (vec (sort (distinct (concat (:source-refs top)
                                                            (:source-refs second-p)))))}))
              top)
            ;; re-scan the recombinant (I-safety)
            merged (assoc merged
                          :charter-ok (scan-ok (proposal-text merged))
                          :review-score (stable-score (proposal-text merged)))]
        (-> state
            (assoc :merged merged)
            (update :datoms conj (datom "shinka:proposal/merged" ":proposal/source" (:pid top)))
            (update :datoms conj (datom "shinka:proposal/merged" ":proposal/charter-ok" (:charter-ok merged))))))))

(defn node-synthesize
  "Meta-review: PR draft (NEVER auto-merge, I2) + dry-run Loop-B corpus feed."
  [state]
  (let [winner (:merged state)
        task (:task state)
        n-kept (count (filterv (complement :is-duplicate) (:proposals state)))
        meta-review (str "Shinka evolution over task " (py-repr-str task) ": "
                         (count (:proposals state)) " charter-clean proposals "
                         "(" (count (:rejected state)) " rejected), " n-kept
                         " contenders debated in " (count (:debates state)) " matches; winner = "
                         (if winner (:pid winner) "none") ".")
        state (assoc state :meta-review meta-review)]
    (if (and (some? winner) (:charter-ok winner))
      (-> state
          (assoc :pr-draft
                 {:title (str "shinka: " (:kind winner) " for " task)
                  :body (str meta-review "\n\n" (:body winner) "\n\nRationale: " (:rationale winner))
                  :member-signed false   ;; I2 — requires a member CACAO capability
                  :auto-merge false      ;; I2 — never autonomous
                  :source-refs (:source-refs winner)})
          ;; Loop-B coupling (DRY-RUN): stage the winner as a Maxwell SFT pair.
          (update :corpus-candidates conj
                  {:id (str "shinka/" task "/" (:pid winner))
                   :instruction (str "Advance task: " task)
                   :completion (:body winner)})
          (update :datoms conj (datom "shinka:pr/draft" ":pr/winner" (:pid winner)))
          (update :datoms conj (datom "shinka:pr/draft" ":pr/auto-merge" false)))
      state)))

;; --------------------------------------------------------------------------- ;;
;; The cell
;; --------------------------------------------------------------------------- ;;

(def ^:private ORDER
  [:propose :reflect :cluster :rank :recombine :synthesize])

(defn make-cell
  "Supervisor-driven generate→debate→evolve→synthesize super-step graph.
  `infer` is an optional Murakumo inference fn (string→string); `sampler` is an
  optional fleet sampler backing `propose`. When both are nil the deterministic
  kernel drives every node (I3 fail-open). Returns an opaque cell map."
  ([] (make-cell nil nil))
  ([infer sampler] {:infer infer :sampler sampler}))

(defn solve
  "Run the full beat for one task (Python `ShinkaEvolutionCell.solve`). No LangGraph
  in cljc — this is the identical sequential super-step driver the Python cell uses
  when StateGraph is None."
  [cell state]
  (-> state
      (node-propose (:sampler cell))
      node-reflect
      node-cluster
      (node-rank (:infer cell))
      node-recombine
      node-synthesize))

(defn is-committable
  "I2: a PR draft becomes committable ONLY with a member CACAO capability."
  [state]
  (boolean
   (and (:pr-draft state)
        (:member-cacao state)
        (not (get-in state [:pr-draft :auto-merge] false)))))
