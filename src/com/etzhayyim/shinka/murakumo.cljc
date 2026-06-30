(ns com.etzhayyim.shinka.murakumo
  "Religious-corp Shinka actor logic migrated from kotoba-kotodama.

  This namespace is the cljc actor boundary for did:web:shinka.etzhayyim.com.
  Side-effecting MST/IPFS/PDS writes are represented as effect maps so the same
  logic can run in JVM, CLJS, or a host actor runtime."
  (:require [clojure.string :as str]))

(def actor-did "did:web:shinka.etzhayyim.com")

(def observe-collection "com.etzhayyim.apps.etzhayyim.shinka.observeAdherent")
(def validate-collection "com.etzhayyim.apps.etzhayyim.shinka.validateEvolution")
(def evolution-collection "com.etzhayyim.apps.etzhayyim.evolutionEvent")
(def heartbeat-collection "com.etzhayyim.apps.etzhayyim.shinka.shinkaHeartbeat")

(def default-axes
  {:joy 40
   :calm 40
   :stress 20
   :gratitude 30
   :focus 40})

(def cadence-minutes
  {:joyful   {:post 30  :engage 15  :drill nil :validate nil :analyze 60}
   :calm     {:post 120 :engage 60  :drill nil :validate 120 :analyze 60}
   :stressed {:post nil :engage nil :drill 30  :validate 60  :analyze nil}
   :grateful {:post 60  :engage 10  :drill nil :validate nil :analyze 60}
   :focused  {:post 180 :engage nil :drill 60  :validate 120 :analyze 30}
   :neutral  {:post 120 :engage 60  :drill 120 :validate 120 :analyze 60}})

(defn clamp
  [lo hi n]
  (-> n (max lo) (min hi)))

(defn normalize-axes
  [axes]
  (reduce-kv
   (fn [m k default]
     (assoc m k (clamp 0 100 (long (get axes k default)))))
   {}
   default-axes))

(defn classify-mood
  "Port of shinka_murakumo._classify_mood."
  [axes]
  (let [{:keys [joy calm stress gratitude focus]} (normalize-axes axes)]
    (cond
      (>= stress 70) :stressed
      (>= joy 60) :joyful
      (>= calm 60) :calm
      (>= gratitude 60) :grateful
      (>= focus 60) :focused
      :else :neutral)))

(defn cadence-flags
  "Resolve shinka actions from mood and elapsed minutes.

  `elapsed-minutes` is the time since the actor's last heartbeat/evolution
  observation. Nil elapsed means no previous heartbeat, so every finite cadence
  is due."
  [{:keys [axes elapsed-minutes]}]
  (let [mood (classify-mood axes)
        policy (cadence-minutes mood)
        due? (fn [k]
               (let [threshold (get policy k)]
                 (boolean (and threshold
                               (or (nil? elapsed-minutes)
                                   (>= elapsed-minutes threshold))))))]
    {:mood mood
     :should-post (due? :post)
     :should-engage (due? :engage)
     :should-drill (due? :drill)
     :should-validate (due? :validate)
     :should-analyze (due? :analyze)}))

(defn safe-rkey
  [s]
  (let [clean (-> (str s)
                  (str/replace #"^did:web:" "")
                  (str/replace #"[^A-Za-z0-9._~-]" "-"))]
    (if (str/blank? clean) "unknown" clean)))

(defn heartbeat-record
  [{:keys [adherent-did axes elapsed-minutes now]}]
  (let [axes* (normalize-axes axes)
        cadence (cadence-flags {:axes axes* :elapsed-minutes elapsed-minutes})]
    {:collection heartbeat-collection
     :rkey (str "heartbeat-" (safe-rkey adherent-did))
     :record (merge
              {:$type heartbeat-collection
               :actorDid adherent-did
               :serviceDid actor-did
               :computedAt now}
              axes*
              cadence)}))

(defn put-record-effect
  [collection rkey record]
  {:op :mst/put-record
   :actor actor-did
   :collection collection
   :rkey rkey
   :record record})

(defn observation-effect
  [adherent]
  (put-record-effect
   observe-collection
   (str "observe-" (safe-rkey (:adherent-did adherent)))
   adherent))

(defn validation-effect
  [claim result]
  (put-record-effect
   validate-collection
   (str "validate-" (safe-rkey (:claim-id claim)))
   {:claim claim
    :result result}))

(defn evolution-effect
  [event]
  (put-record-effect
   evolution-collection
   (str "evolution-" (safe-rkey (:event-id event)))
   event))

(defn heartbeat-effect
  [input]
  (let [{:keys [collection rkey record]} (heartbeat-record input)]
    (put-record-effect collection rkey record)))

(defn observation-cell-plan
  "Plan for legacy karma_hegemon_observation_cell.

  The host is responsible for fetching kyumei signals and any pending claim.
  This pure plan normalizes axes/cadence and emits the MST observation write."
  [{:keys [adherent-did axes elapsed-minutes now kyumei-signals proposed-evolution]}]
  (let [axes* (normalize-axes axes)
        cadence (cadence-flags {:axes axes* :elapsed-minutes elapsed-minutes})
        adherent {:adherent-did adherent-did
                  :axes axes*
                  :kyumei-signals (vec (or kyumei-signals []))
                  :proposed-evolution proposed-evolution
                  :computed-at now
                  :cadence cadence}]
    {:cell :karma-hegemon-observation
     :adherent adherent
     :effects [(observation-effect adherent)]}))

(defn validation-cell-plan
  "Plan for legacy evolution_validation_cell."
  [{:keys [claim result now]}]
  (let [result* (merge {:validated false
                        :reason "pending-host-validation"
                        :validated-at now}
                       (or result {}))]
    {:cell :evolution-validation
     :claim claim
     :result result*
     :effects [(validation-effect claim result*)]}))

(defn emission-cell-plan
  "Plan for legacy evolution_emission_cell."
  [{:keys [event now]}]
  (let [event* (merge {:emitted-at now
                       :serviceDid actor-did}
                      event)]
    {:cell :evolution-emission
     :event event*
     :effects [(evolution-effect event*)]}))

(defn heartbeat-cell-plan
  "Plan for legacy shinka_heartbeat_cell."
  [input]
  {:cell :shinka-heartbeat
   :effects [(heartbeat-effect input)]})

(defn tick-plan
  "High-level pure plan for legacy shinka_tick.

  Produces the observation and heartbeat effects every tick. Validation and
  emission are added only when the host supplies a proposed evolution claim."
  [{:keys [proposed-evolution] :as input}]
  (let [observation (observation-cell-plan input)
        heartbeat (heartbeat-cell-plan input)
        validation (when proposed-evolution
                     (validation-cell-plan {:claim proposed-evolution
                                            :now (:now input)}))
        emission (when proposed-evolution
                   (emission-cell-plan {:event {:event-id (:claim-id proposed-evolution)
                                                :adherentDid (:adherent-did input)
                                                :claim proposed-evolution}
                                        :now (:now input)}))]
    {:cell :shinka-tick
     :effects (vec (mapcat :effects (remove nil? [observation heartbeat validation emission])))
     :steps (vec (remove nil? [observation heartbeat validation emission]))}))
