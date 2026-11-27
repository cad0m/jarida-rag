import { MessageSquare, PenSquare, Settings as SettingsIcon, Trash2, X } from "lucide-react"
import { AnimatePresence, motion } from "motion/react"

function Logo() {
  return (
    <div className="logo">
      <img className="logo-mark" src="/assets/medallion.webp" alt="" />
      <span className="logo-words">
        <span className="logo-name">Jarida</span>
        <span className="logo-arabic" lang="ar">الجريدة الرسمية</span>
      </span>
    </div>
  )
}

export default function Sidebar({ chats, activeId, open, onClose, onNew, onSelect, onDelete, onSettings }) {
  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.div
            className="scrim"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
        )}
      </AnimatePresence>

      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="sidebar-top">
          <Logo />
          <button className="icon-button mobile-only" onClick={onClose} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <motion.button className="button primary new-chat" onClick={onNew} whileTap={{ scale: 0.97 }}>
          <PenSquare size={16} />
          New chat
        </motion.button>

        <p className="sidebar-label">Chats</p>
        <nav className="chat-list">
          {chats.length === 0 && <p className="sidebar-empty">Your questions will show up here.</p>}
          <AnimatePresence initial={false}>
            {chats.map((chat) => (
              <motion.div
                key={chat.id}
                layout
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, height: 0 }}
                className={`chat-item ${chat.id === activeId ? "active" : ""}`}
              >
                <button className="chat-title" onClick={() => onSelect(chat.id)} dir="auto">
                  <MessageSquare size={15} />
                  <span>{chat.title}</span>
                </button>
                <button className="icon-button delete" onClick={() => onDelete(chat.id)} aria-label="Delete chat">
                  <Trash2 size={14} />
                </button>
              </motion.div>
            ))}
          </AnimatePresence>
        </nav>

        <div className="sidebar-band" aria-hidden="true" />
        <button className="sidebar-settings" onClick={onSettings}>
          <SettingsIcon size={16} />
          Settings
        </button>
      </aside>
    </>
  )
}
