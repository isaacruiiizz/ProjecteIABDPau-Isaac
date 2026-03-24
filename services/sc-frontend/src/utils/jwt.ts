/**
 * Decodifica el payload d'un JWT (sense verificació de signatura).
 * Retorna el camp `role` si existeix, o null.
 */
export function parseJwtRole(token: string): string | null {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return (payload.role as string) ?? null;
  } catch {
    return null;
  }
}
