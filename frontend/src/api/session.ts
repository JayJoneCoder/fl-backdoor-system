// Session cookies are HttpOnly. Only the per-session CSRF value lives in memory.
export { setCsrfToken, getCsrfToken, notifySessionExpired } from './checkedFetch';