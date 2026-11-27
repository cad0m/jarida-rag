import { motion, useReducedMotion } from "motion/react"
import Composer, { Chips, Notices } from "./Composer.jsx"

const CAMERA = [0.16, 1, 0.3, 1]
const LEAVE = [0.7, 0, 0.84, 0]
const MOTES = 10

// The court's doorway, as layers: far hall (behind, in Chat), sunlight, arcades, the gate, notices, lanterns.
// The camera only moves on purpose: arriving (intro), coming back (return) and walking through (exit).
// Each layer travels by its own amount, so the move reads in depth; the idle life is CSS loops (styles.css).
// Positions inside .gate are fractions of the gate image (the arch's tip is at 26% of its height).
export default function Hero({ arrival, narrow, onSend, loading }) {
  const reduceMotion = useReducedMotion()
  const move = reduceMotion ? "still" : arrival
  const pace = move === "intro" ? 1 : 0.45
  const from = (intro, back) => (move === "intro" ? intro : move === "return" ? back : false)
  const camera = (delay = 0, duration = 1.8) => ({ duration: duration * pace, delay: delay * pace, ease: CAMERA })

  return (
    <motion.div className="doorway" exit={{ opacity: 0, transition: { duration: 0.5, delay: 0.45 } }}>
      <motion.div
        className="layer sunlight"
        aria-hidden="true"
        initial={from({ opacity: 0 }, { opacity: 0 })}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0, transition: { duration: 0.3 } }}
        transition={camera(1, 1.2)}
      >
        <span className="beam" />
        <span className="beam" />
        <span className="beam" />
        <div className="motes">
          {Array.from({ length: MOTES }, (_, i) => (
            <span key={i} style={{ "--x": `${8 + ((i * 37) % 84)}%`, "--d": `${-i * 1.7}s`, "--t": `${11 + (i % 4) * 3}s` }} />
          ))}
        </div>
      </motion.div>

      <div className="layer arcades" aria-hidden="true">
        {["left", "right"].map((side) => {
          const out = side === "left" ? -1 : 1
          return (
            <motion.img
              key={side}
              className={`arcade ${side}`}
              src={`/assets/arcade-${side}.webp`}
              alt=""
              draggable="false"
              initial={from({ x: `${-out * 45}%`, scale: 0.6, opacity: 0 }, { x: `${out * 40}%`, scale: 1.4, opacity: 0 })}
              animate={{ x: "0%", scale: 1, opacity: 1 }}
              exit={{ x: `${out * 70}%`, scale: 1.5, opacity: 0, transition: { duration: 0.8, ease: LEAVE } }}
              transition={camera(0.1, 1.9)}
            />
          )
        })}
      </div>

      <div className="layer gate-layer">
        <div className="gate-anchor">
          <motion.div
            className="gate"
            initial={from({ scale: 0.34, y: "12%", opacity: 0 }, { scale: 1.6, opacity: 0 })}
            animate={{ scale: 1, y: "0%", opacity: 1 }}
            exit={{ scale: 2.4, opacity: 0, transition: { duration: 0.9, ease: LEAVE } }}
            transition={{ ...camera(0, 1.8), opacity: { duration: 0.6 * pace } }}
          >
            <img src="/assets/gate.webp" alt="" draggable="false" fetchPriority="high" />
            <span className="inscription" lang="ar">العدل أساس الملك</span>
            <div className="door-light" aria-hidden="true" />

            <motion.div className="door-copy" initial={from({ opacity: 0 }, { opacity: 0 })} animate={{ opacity: 1 }} transition={camera(1, 0.7)}>
              <p className="eyebrow">Bulletin Officiel · an unofficial reader</p>
              <motion.h1
                className="masthead"
                lang="ar"
                dir="rtl"
                initial={from({ clipPath: "inset(0 0 0 100%)" }, false)}
                animate={{ clipPath: "inset(0 0 0 0%)" }}
                transition={{ duration: 1, delay: 1.1, ease: [0.65, 0, 0.35, 1] }}
              >
                الجريدة الرسمية
              </motion.h1>
              <p className="tagline">Ask the law, in its own words.</p>
            </motion.div>

            {!narrow && (
              <motion.div className="door-desk" initial={from({ opacity: 0, y: 14 }, { opacity: 0, y: 8 })} animate={{ opacity: 1, y: 0 }} transition={camera(1.5, 0.6)}>
                <Composer onSend={onSend} loading={loading} />
                <Chips onSend={onSend} className="door-chips" />
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>

      {!narrow && (
        <div className="layer walls">
          <Notices onSend={onSend} delay={move === "still" ? null : 1.7 * pace} />
        </div>
      )}

      <div className="layer lanterns" aria-hidden="true">
        {["left", "right"].map((side, index) => (
          <motion.div
            key={side}
            className={`lantern ${side}`}
            initial={from({ y: "-120%" }, { y: "-120%" })}
            animate={{ y: "0%" }}
            exit={{ y: "-130%", transition: { duration: 0.6, ease: LEAVE } }}
            transition={{ type: "spring", stiffness: 60, damping: 9, delay: (1.3 + index * 0.2) * pace }}
          >
            <span className="lantern-glow" />
            <img className="sway" src="/assets/lantern.webp" alt="" draggable="false" />
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}
