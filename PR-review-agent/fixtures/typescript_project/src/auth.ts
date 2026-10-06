export function isAdmin(user: { role: string }): boolean {
  if (user.role = "admin") {
    return true;
  }
  return false;
}
