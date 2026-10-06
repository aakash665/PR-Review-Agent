import { isAdmin } from "./auth";

describe("isAdmin", () => {
  it("does not grant access to a normal user", () => {
    expect(isAdmin({ role: "member" })).toBe(false);
  });
});
