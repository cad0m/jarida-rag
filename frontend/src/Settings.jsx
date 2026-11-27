import { X } from "lucide-react"
import { AnimatePresence, motion } from "motion/react"
import { useEffect, useState } from "react"
import { isOnline } from "./api.js"

const TABS = ["General", "Connection", "Data", "About"]
const THEMES = [
  ["system", "System"],
  ["light", "Light"],
  ["dark", "Dark"],
]

export default function Settings({ open, onClose, settings, onChange, chatCount, onDeleteChats }) {
  const [tab, setTab] = useState("General")

  useEffect(() => {
    const closeOnEscape = (event) => event.key === "Escape" && onClose()
    window.addEventListener("keydown", closeOnEscape)
    return () => window.removeEventListener("keydown", closeOnEscape)
  }, [onClose])

  return (
    <AnimatePresence>
      {open && (
        <motion.div className="modal-backdrop" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <motion.div
            className="modal"
            role="dialog"
            aria-label="Settings"
            onClick={(event) => event.stopPropagation()}
            initial={{ opacity: 0, scale: 0.96, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 16 }}
            transition={{ type: "spring", stiffness: 380, damping: 32 }}
          >
            <header className="modal-head">
              <h2>Settings</h2>
              <button className="icon-button" onClick={onClose} aria-label="Close settings">
                <X size={18} />
              </button>
            </header>

            <div className="tabs" role="tablist">
              {TABS.map((name) => (
                <button key={name} role="tab" aria-selected={tab === name} className="tab" onClick={() => setTab(name)}>
                  {tab === name && <motion.span layoutId="tab-pill" className="tab-pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
                  <span>{name}</span>
                </button>
              ))}
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={tab}
                className="tab-body"
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.18 }}
              >
                {tab === "General" && <General settings={settings} onChange={onChange} />}
                {tab === "Connection" && <Connection settings={settings} onChange={onChange} />}
                {tab === "Data" && <Data chatCount={chatCount} onDeleteChats={onDeleteChats} />}
                {tab === "About" && <About />}
              </motion.div>
            </AnimatePresence>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function Row({ title, hint, children }) {
  return (
    <div className="row">
      <div>
        <p className="row-title">{title}</p>
        {hint && <p className="row-hint">{hint}</p>}
      </div>
      {children}
    </div>
  )
}

function General({ settings, onChange }) {
  return (
    <Row title="Theme" hint="System follows your device.">
      <div className="segmented">
        {THEMES.map(([value, label]) => (
          <button key={value} onClick={() => onChange({ theme: value })}>
            {settings.theme === value && <motion.span layoutId="theme-pill" className="segment-pill" />}
            <span>{label}</span>
          </button>
        ))}
      </div>
    </Row>
  )
}

function Connection({ settings, onChange }) {
  const [state, setState] = useState("idle")

  async function test() {
    setState("testing")
    setState((await isOnline(settings.apiUrl)) ? "online" : "offline")
  }

  return (
    <>
      <Row title="API URL" hint="Where the RAG answers: the /ask and /health endpoints." />
      <div className="field">
        <input
          value={settings.apiUrl}
          placeholder="https://your-api.example.com"
          onChange={(event) => {
            onChange({ apiUrl: event.target.value })
            setState("idle")
          }}
        />
        <motion.button className="button primary" onClick={test} whileTap={{ scale: 0.96 }} disabled={state === "testing"}>
          {state === "testing" ? "Testing…" : "Test"}
        </motion.button>
      </div>
      {state === "online" && <p className="note ok">Connected. The API answered /health.</p>}
      {state === "offline" && <p className="note bad">No answer from /health. Check the URL and that the API is running.</p>}
    </>
  )
}

function Data({ chatCount, onDeleteChats }) {
  const [confirming, setConfirming] = useState(false)
  return (
    <Row title="Chat history" hint={`${chatCount} chat${chatCount === 1 ? "" : "s"}, saved in this browser only.`}>
      <button
        className={`button ${confirming ? "danger" : "secondary"}`}
        disabled={chatCount === 0}
        onClick={() => {
          if (confirming) onDeleteChats()
          setConfirming(!confirming)
        }}
      >
        {confirming ? "Click to confirm" : "Delete all"}
      </button>
    </Row>
  )
}

function About() {
  return (
    <div className="about">
      <p>
        Jarida answers questions about the Moroccan Official Gazette (الجريدة الرسمية) from the gazette itself: it finds the
        relevant articles in the published issues and cites each one.
      </p>
      <p className="row-hint">
        Issues come from sgg.gov.ma. Retrieval uses bge-m3 and a reranker over Postgres with pgvector; answers are written by
        Qwen3. Always check the cited issue before relying on an answer.
      </p>
    </div>
  )
}
