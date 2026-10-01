export const PASSWORD_REQUIREMENTS = [
  { id: "length", label: "At least 8 characters", test: (value: string) => value.length >= 8 },
  { id: "uppercase", label: "At least 1 uppercase letter", test: (value: string) => /[A-Z]/.test(value) },
  { id: "lowercase", label: "At least 1 lowercase letter", test: (value: string) => /[a-z]/.test(value) },
  { id: "number", label: "At least 1 number", test: (value: string) => /\d/.test(value) },
  { id: "special", label: "At least 1 special character", test: (value: string) => /[^A-Za-z0-9]/.test(value) },
] as const;

export function passwordChecks(value: string): Record<string, boolean> {
  return Object.fromEntries(PASSWORD_REQUIREMENTS.map((rule) => [rule.id, rule.test(value)]));
}

export function isStrongPassword(value: string): boolean {
  return PASSWORD_REQUIREMENTS.every((rule) => rule.test(value));
}

export function passwordStrengthLabel(value: string): "Weak" | "Almost there" | "Strong" {
  const passed = PASSWORD_REQUIREMENTS.filter((rule) => rule.test(value)).length;
  if (passed >= 5) return "Strong";
  if (passed >= 3) return "Almost there";
  return "Weak";
}
