import { ArrowUp } from "lucide-react"
import { motion } from "motion/react"
import { useState } from "react"

export const SUGGESTIONS = [
  "ما هي شروط إنشاء الجماعات المحلية؟",
  "Which law regulates municipalities?",
  "What did decree 2.24.830 change?",
  "ما هي آخر المراسيم المتعلقة بالمالية؟",
]

export default function Composer({ onSend, loading, className = "" }) {
  const [text, setText] = useState("")
  const canSend = text.trim() && !loading

  function submit(event) {
    event.preventDefault()
    if (!canSend) return
    onSend(text.trim())
    setText("")
  }

  return (
    <motion.form
      layoutId="composer"
      className={`composer ${className}`}
      onSubmit={submit}
      transition={{ type: "spring", stiffness: 260, damping: 30 }}
    >
      <textarea
        id="composer-input"
        value={text}
        dir="auto"
        rows={1}
        aria-label="Your question"
        placeholder="Ask a legal question · اطرح سؤالك"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) submit(event)
        }}
      />
      <kbd className="hint">/</kbd>
      <motion.button type="submit" className="send" disabled={!canSend} whileTap={{ scale: 0.9 }} aria-label="Send">
        <ArrowUp size={18} />
      </motion.button>
    </motion.form>
  )
}

// court notices pinned to the walls on each side of the gate
export function Notices({ onSend, delay }) {
  const sides = [SUGGESTIONS.slice(0, 2), SUGGESTIONS.slice(2)]
  return sides.map((questions, side) => (
    <div key={side} className={`notices ${side ? "right" : "left"}`}>
      {questions.map((question, index) => {
        const tilt = [-3, 2, 2.5, -2][side * 2 + index]
        return (
          <motion.button
            key={question}
            className="notice"
            onClick={() => onSend(question)}
            initial={delay === null ? false : { opacity: 0, scale: 1.15, rotate: tilt * 3, y: -20 }}
            animate={{ opacity: 1, scale: 1, rotate: tilt, y: 0 }}
            transition={{ type: "spring", stiffness: 180, damping: 16, delay: (delay ?? 0) + (side * 2 + index) * 0.12 }}
            whileHover={{ rotate: 0, y: -4, scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            <span className="pin" aria-hidden="true" />
            <span className="notice-head">
              <span>Avis</span>
              <span lang="ar">إعلان</span>
            </span>
            <span className="notice-text" dir="auto">
              {question}
            </span>
          </motion.button>
        )
      })}
    </div>
  ))
}

export function Chips({ onSend, className = "" }) {
  return (
    <div className={`chips ${className}`}>
      {SUGGESTIONS.map((question) => (
        <motion.button key={question} className="pill" dir="auto" whileTap={{ scale: 0.97 }} onClick={() => onSend(question)}>
          {question}
        </motion.button>
      ))}
    </div>
  )
}
