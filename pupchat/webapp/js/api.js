/* Every request the page makes, in one place.
 *
 * One module knows the endpoints and one shape of error comes back, so a view never
 * has to decide what a 403 means or whether `detail` exists this time.
 *
 * The browser never holds an API key. It talks only to this server, which keeps the key
 * in .env and never passes it on.
 */

const BASE = ""; // same origin

export class ApiError extends Error {
  constructor(message, status, retryAfter) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

/* The server's own words wherever it gave any.
 *
 * Its messages are written for the owner - "that clip is 12.2s, Pupchat needs 5-10s" -
 * and are far more useful than anything this layer could invent from a status code. */
async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, options);
  } catch (e) {
    throw new ApiError("Pupchat is not answering. Is the server window still open?", 0);
  }
  if (res.status === 204) return null;

  const isJson = (res.headers.get("content-type") || "").includes("json");
  const body = isJson ? await res.json().catch(() => null) : null;

  if (!res.ok) {
    throw new ApiError(
      body?.detail || `Something went wrong (${res.status}).`,
      res.status,
      Number(res.headers.get("retry-after")) || 0
    );
  }
  return body;
}

const json = (method, path, data) =>
  request(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data ?? {}),
  });

export const api = {
  health: () => request("/api/health"),
  library: () => request("/api/library"),

  createDog: (data) => json("POST", "/api/dogs", data),
  updateDog: (id, data) => json("PUT", `/api/dogs/${id}`, data),
  deleteDog: (id) => request(`/api/dogs/${id}`, { method: "DELETE" }),
  setActiveDog: (dogId) => json("PUT", "/api/active-dog", { dog_id: dogId }),

  uploadPhoto: (id, file) => {
    const fd = new FormData();
    fd.append("file", file);
    return request(`/api/dogs/${id}/photo`, { method: "PUT", body: fd });
  },
  photoUrl: (dog) =>
    // The cache-buster is the dog's own updated stamp, so replacing a photo shows the
    // new one immediately while an unchanged one still caches.
    dog?.photo ? `/api/dogs/${dog.id}/photo?v=${Math.round(dog.updated || 0)}` : null,

  addBaseline: (id, entry) => json("POST", `/api/dogs/${id}/baselines`, entry),
  removeBaseline: (id, hash) =>
    request(`/api/dogs/${id}/baselines/${hash}`, { method: "DELETE" }),

  createChat: (dogId, title) => json("POST", "/api/chats", { dog_id: dogId, title }),
  chat: (id) => request(`/api/chats/${id}`),
  patchChat: (id, data) => json("PATCH", `/api/chats/${id}`, data),
  deleteChat: (id) => request(`/api/chats/${id}`, { method: "DELETE" }),

  videoUrl: (hash) => `/api/video/${hash}`,
};
