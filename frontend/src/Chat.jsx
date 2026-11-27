import { Check, Copy, ExternalLink, Menu, RotateCcw } from "lucide-react"
import { AnimatePresence, motion, useReducedMotion } from "motion/react"
import { useEffect, useRef, useState } from "react"
import Composer, { Chips } from "./Composer.jsx"
import Hero from "./Hero.jsx"

export default function Chat({ messages, loading, online, intro, onSend, onRetry, onMenu, onSettings }) {
  const bottom = useRef(null)
  const empty = messages.length === 0
  const narrow = useNarrow()
  const reduceMotion = useReducedMotion()

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages.length, loading])

  return (
    <main className={`chat ${empty ? "at-door" : "inside"}`}>
      {/* the hall is the room for the whole session: at the door it is the view, inside it is a blurred
          backdrop; both versions are baked images, so switching is a cheap fade, not a live blur */}
      <div className="backdrop" aria-hidden="true">
        <div className={`drift ${empty ? "" : "paused"}`}>
          <motion.div
            className="hall"
            initial={intro && !reduceMotion ? { scale: 1 } : false}
            animate={{ scale: empty ? 1.1 : 1.2 }}
            transition={{ duration: empty ? 1.8 : 1.1, ease: [0.16, 1, 0.3, 1] }}
          />
        </div>
        <motion.div className="hall room" initial={false} animate={{ opacity: empty ? 0 : 1 }} transition={{ duration: 0.8, delay: empty ? 0 : 0.3 }} />
      </div>
      <motion.div className="veil" initial={false} animate={{ opacity: empty ? 0 : 1 }} transition={{ duration: 0.8, delay: empty ? 0 : 0.3 }} />

      <header className="topbar">
        <button className="icon-button mobile-only" onClick={onMenu} aria-label="Open menu">
          <Menu size={20} />
        </button>
        <button className={`status ${online ? "online" : ""}`} onClick={onSettings}>
          <span className="dot" />
          {online ? "Connected" : "Not connected"}
        </button>
      </header>

      <AnimatePresence>
        {empty && <Hero key="doorway" arrival={intro ? "intro" : "return"} narrow={narrow} onSend={onSend} loading={loading} />}
      </AnimatePresence>

      {empty ? (
        <>
          <div className="spacer" />
          {narrow && (
            <div className="mobile-dock">
              <Chips onSend={onSend} />
              <Composer onSend={onSend} loading={loading} />
            </div>
          )}
        </>
      ) : (
        <>
          {/* waits for the walk through the gate to pass before the conversation appears */}
          <motion.section className="messages" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.55, duration: 0.4 }}>
            {messages.map((message, index) => (
              <Message key={index} message={message} onRetry={() => onRetry(index)} />
            ))}
            <AnimatePresence>{loading && <Thinking />}</AnimatePresence>
            <div ref={bottom} />
          </motion.section>
          <div className="composer-dock">
            <Composer onSend={onSend} loading={loading} />
          </div>
        </>
      )}
      <p className="disclaimer">Unofficial assistant. Verify every answer in the Bulletin Officiel on sgg.gov.ma.</p>
    </main>
  )
}

function useNarrow() {
  const query = "(max-width: 820px)"
  const [narrow, setNarrow] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const list = window.matchMedia(query)
    const change = () => setNarrow(list.matches)
    list.addEventListener("change", change)
    return () => list.removeEventListener("change", change)
  }, [])
  return narrow
}

function Message({ message, onRetry }) {
  const [selected, setSelected] = useState(null)
  const sources = message.sources ?? []
  const assistant = message.role === "assistant"

  return (
    <motion.article
      className={`message ${message.role} ${message.error ? "error" : ""}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      {assistant && !message.error && <img className="avatar" src="/assets/medallion.webp" alt="" />}
      <div className="message-body">
        <div className="bubble" dir="auto">
          {assistant ? withCitations(message.text, sources, setSelected) : message.text}
        </div>

        {message.error && message.question && (
          <button className="button secondary small" onClick={onRetry}>
            <RotateCcw size={14} />
            Try again
          </button>
        )}

        {sources.length > 0 && (
          <div className="sources">
            {sources.map((source, index) => (
              <Clipping
                key={index}
                source={source}
                index={index}
                active={selected === index}
                onSelect={() => setSelected(selected === index ? null : index)}
              />
            ))}
          </div>
        )}

        <AnimatePresence>
          {selected !== null && sources[selected] && <Excerpt key={selected} source={sources[selected]} />}
        </AnimatePresence>

        {assistant && !message.error && <CopyButton text={message.text} />}
      </div>
    </motion.article>
  )
}

function sourceIndex(citation, sources) {
  const issue = citation.match(/BO\s*([\w-]+)/)?.[1]
  const page = citation.match(/p\.\s*(\d+)/)?.[1]
  const exact = sources.findIndex((source) => String(source.issue) === issue && String(source.page_start) === page)
  return exact >= 0 ? exact : sources.findIndex((source) => String(source.issue) === issue)
}

function withCitations(text, sources, select) {
  // the model cites like [BO 7470, 2.24.830, المادة 5, p. 8]: a tag that opens its source below
  return text.split(/(\[BO [^\]]+\])/g).map((part, index) => {
    if (!part.startsWith("[BO ")) return part
    const citation = part.slice(1, -1)
    const target = sourceIndex(citation, sources)
    return target >= 0 ? (
      <button key={index} className="cite" dir="auto" onClick={() => select(target)}>
        § {citation}
      </button>
    ) : (
      <span key={index} className="cite" dir="auto">
        § {citation}
      </span>
    )
  })
}

function Clipping({ source, index, active, onSelect }) {
  return (
    <motion.button
      className={`clipping ${active ? "active" : ""}`}
      onClick={onSelect}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 + index * 0.06, type: "spring", stiffness: 200, damping: 22 }}
      whileHover={{ y: -3 }}
    >
      <span className="clipping-masthead">
        <span>Bulletin Officiel</span>
        <span lang="ar">العدد {source.issue}</span>
      </span>
      <span className="clipping-title" dir="auto">
        {source.title}
      </span>
      <span className="clipping-meta">
        {[source.text_number, source.article && `المادة ${source.article}`, source.page_start && `p. ${source.page_start}`, source.date]
          .filter(Boolean)
          .join(" · ")}
      </span>
    </motion.button>
  )
}

function Excerpt({ source }) {
  const pdf = source.drive_id ? `https://drive.google.com/file/d/${source.drive_id}/view` : null
  return (
    <motion.div
      className="excerpt"
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="excerpt-inner">
        <p className="excerpt-text" dir="auto">
          {source.text || "The API didn't send the text of this source."}
        </p>
        {pdf && (
          <a className="excerpt-link" href={pdf} target="_blank" rel="noreferrer">
            Open issue {source.issue}
            {source.page_start ? `, page ${source.page_start}` : ""}
            <ExternalLink size={13} />
          </a>
        )}
      </div>
    </motion.div>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="copy"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        })
      }}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
      {copied ? "Copied" : "Copy"}
    </button>
  )
}

function Thinking() {
  return (
    <motion.div className="thinking" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
      <motion.img
        className="avatar"
        src="/assets/medallion.webp"
        alt=""
        animate={{ rotate: 360 }}
        transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
      />
      <span>Searching the gazette archive…</span>
    </motion.div>
  )
}
