import { usePersonalization } from "./PersonalizationContext";

export default function ProfileSwitcher() {
  const { profiles, activeProfile, switchProfile, loading, saving } = usePersonalization();

  if (loading || !activeProfile) {
    return <span className="profile-switcher loading">Loading profile…</span>;
  }

  return (
    <label className="profile-switcher">
      <span className="profile-switcher-label">Profile</span>
      <select
        value={activeProfile.id}
        onChange={(event) => void switchProfile(Number(event.target.value))}
        disabled={saving}
        aria-label="Active communication profile"
      >
        {profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}
      </select>
    </label>
  );
}
