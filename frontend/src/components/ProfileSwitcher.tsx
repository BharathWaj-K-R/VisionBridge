import { useNavigate } from "react-router-dom";
import { usePersonalization } from "./PersonalizationContext";

const CREATE_PROFILE_VALUE = "__create_profile__";

export default function ProfileSwitcher() {
  const { profiles, activeProfile, switchProfile, loading, saving } = usePersonalization();
  const navigate = useNavigate();

  if (loading || !activeProfile) {
    return <span className="profile-switcher loading">Loading profile…</span>;
  }

  return (
    <label className="profile-switcher">
      <span className="profile-switcher-label">Communication profile</span>
      <select
        value={activeProfile.id}
        onChange={(event) => {
          if (event.target.value === CREATE_PROFILE_VALUE) {
            event.target.value = String(activeProfile.id);
            navigate("/personalization?create=1");
            return;
          }
          void switchProfile(Number(event.target.value));
        }}
        disabled={saving}
        aria-label="Active communication profile"
      >
        {profiles.map((profile) => (
          <option key={profile.id} value={profile.id}>{profile.name}</option>
        ))}
        <option value={CREATE_PROFILE_VALUE}>+ Create profile</option>
      </select>
    </label>
  );
}
