// The RAG API (see README): GET /health, POST /ask {question} -> {answer, sources}

function base(apiUrl) {
  return apiUrl.trim().replace(/\/+$/, "")
}

export async function ask(apiUrl, question) {
  const response = await fetch(`${base(apiUrl)}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  })
  if (!response.ok) throw new Error(`The API answered with status ${response.status}.`)
  return response.json()
}

export async function isOnline(apiUrl) {
  if (!apiUrl.trim()) return false
  try {
    return (await fetch(`${base(apiUrl)}/health`)).ok
  } catch {
    return false
  }
}
