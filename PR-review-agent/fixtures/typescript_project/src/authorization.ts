export function hasRole(user: { role: string }, role: string): boolean {
  return user.role === role;
}
