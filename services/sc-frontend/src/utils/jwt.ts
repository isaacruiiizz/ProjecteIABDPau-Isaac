/**
 * Decodifica el payload d'un JWT (sense verificació de signatura).
 * Retorna el camp `role` si existeix, o null.
 */
export function parseJwtRole(token: string): string | null {
  try {
    const b64 = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(b64));
    return (payload.role as string) ?? null;
  } catch {
    return null;
  }
}
