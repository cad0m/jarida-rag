import { useEffect, useState } from "react"
import { ask, isOnline } from "./api.js"
import Chat from "./Chat.jsx"
import Settings from "./Settings.jsx"
import Sidebar from "./Sidebar.jsx"

const DEFAULT_SETTINGS = { apiUrl: import.meta.env.VITE_API_URL ?? "", theme: "system" }

function useStored(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(key)) ?? initial
    } catch {
      return initial
    }
  })
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // private mode: the app still works, it just forgets on reload
    }
  }, [key, value])
  return [value, setValue]
}

export default function App() {
  const [chats, setChats] = useStored("jarida.chats", [])
  const [settings, setSettings] = useStored("jarida.settings", DEFAULT_SETTINGS)
  const [activeId, setActiveId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [online, setOnline] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  // the opening animation plays once per visit, not on every new chat
  const [intro, setIntro] = useState(() => {
    try {
      return !sessionStorage.getItem("jarida.intro")
    } catch {
      return true
    }
  })

  const activeChat = chats.find((chat) => chat.id === activeId)

  useEffect(() => {
    try {
      sessionStorage.setItem("jarida.intro", "seen")
    } catch {
      // private mode: the intro just plays again next time
    }
  }, [])

  useEffect(() => {
    function onKey(event) {
      const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault()
        startNewChat()
      } else if (event.key === "/" && !typing) {
        event.preventDefault()
        document.getElementById("composer-input")?.focus()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  useEffect(() => {
    if (settings.theme === "system") delete document.documentElement.dataset.theme
    else document.documentElement.dataset.theme = settings.theme
  }, [settings.theme])

  useEffect(() => {
    isOnline(settings.apiUrl).then(setOnline)
  }, [settings.apiUrl])

  function addMessage(chatId, message) {
    setChats((all) => all.map((chat) => (chat.id === chatId ? { ...chat, messages: [...chat.messages, message] } : chat)))
  }

  function send(question) {
    setIntro(false) // from now on the doorway is a return, not a first arrival
    let chatId = activeId
    if (!activeChat) {
      chatId = crypto.randomUUID()
      setChats((all) => [{ id: chatId, title: question.slice(0, 60), messages: [] }, ...all])
      setActiveId(chatId)
    }
    addMessage(chatId, { role: "user", text: question })
    answer(chatId, question)
  }

  function retry(index) {
    const { question } = activeChat.messages[index]
    setChats((all) =>
      all.map((chat) => (chat.id === activeId ? { ...chat, messages: chat.messages.filter((_, i) => i !== index) } : chat)),
    )
    answer(activeId, question)
  }

  async function answer(chatId, question) {
    if (!settings.apiUrl.trim()) {
      addMessage(chatId, { role: "assistant", error: true, question, text: "No API is connected yet. Add its URL in Settings → Connection." })
      return
    }
    setLoading(true)
    try {
      const { answer, sources } = await ask(settings.apiUrl, question)
      addMessage(chatId, { role: "assistant", text: answer, sources: sources ?? [] })
    } catch (error) {
      addMessage(chatId, { role: "assistant", error: true, question, text: `Couldn't reach the gazette API. ${error.message}` })
    } finally {
      setLoading(false)
    }
  }

  function startNewChat() {
    setActiveId(null)
    setSidebarOpen(false)
  }

  function deleteChat(chatId) {
    setChats((all) => all.filter((chat) => chat.id !== chatId))
    if (chatId === activeId) setActiveId(null)
  }

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        activeId={activeId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNew={startNewChat}
        onSelect={(chatId) => {
          setActiveId(chatId)
          setSidebarOpen(false)
        }}
        onDelete={deleteChat}
        onSettings={() => setSettingsOpen(true)}
      />
      <Chat
        messages={activeChat?.messages ?? []}
        loading={loading}
        online={online}
        intro={intro}
        onSend={send}
        onRetry={retry}
        onMenu={() => setSidebarOpen(true)}
        onSettings={() => setSettingsOpen(true)}
      />
      <Settings
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={(changes) => setSettings((current) => ({ ...current, ...changes }))}
        chatCount={chats.length}
        onDeleteChats={() => {
          setChats([])
          setActiveId(null)
        }}
      />
    </div>
  )
}
