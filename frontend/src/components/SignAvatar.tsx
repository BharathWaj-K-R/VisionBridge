import type { AvatarPreferences } from "./AvatarCustomizer";

const POSES: Record<string, { left: number; right: number; spread: number; lift: number }> = {
  A:{left:-12,right:12,spread:2,lift:4},B:{left:-28,right:28,spread:9,lift:0},C:{left:-42,right:42,spread:18,lift:2},
  D:{left:-56,right:56,spread:5,lift:-5},E:{left:-68,right:68,spread:11,lift:5},F:{left:-82,right:82,spread:20,lift:-2},
  G:{left:-38,right:38,spread:3,lift:-11},H:{left:-22,right:22,spread:14,lift:-11},I:{left:8,right:-8,spread:8,lift:-8},
  J:{left:18,right:-18,spread:16,lift:3},K:{left:-48,right:48,spread:6,lift:-7},L:{left:-63,right:63,spread:15,lift:-8},
  M:{left:-76,right:76,spread:4,lift:6},N:{left:-88,right:88,spread:7,lift:3},O:{left:-102,right:102,spread:19,lift:1},
  P:{left:-55,right:55,spread:9,lift:12},Q:{left:-42,right:42,spread:12,lift:13},R:{left:-31,right:31,spread:6,lift:10},
  S:{left:-12,right:12,spread:1,lift:10},T:{left:-95,right:95,spread:5,lift:8},U:{left:-26,right:26,spread:10,lift:-12},
  V:{left:-48,right:48,spread:24,lift:-15},W:{left:-65,right:65,spread:28,lift:-12},X:{left:-74,right:74,spread:13,lift:-16},
  Y:{left:20,right:-20,spread:22,lift:-3},Z:{left:88,right:-88,spread:4,lift:8},
};

export default function SignAvatar({
  letter,
  preferences,
  playing,
}: {
  letter: string;
  preferences: AvatarPreferences;
  playing: boolean;
}) {
  const pose = POSES[letter] || POSES.A;
  const isSpace = letter === " ";

  return (
    <div className={playing ? "sign-avatar playing" : "sign-avatar"} data-letter={letter}>
      <div className="avatar-stage-label">
        <span className="eyebrow">CURRENT SIGN</span>
        <strong>{isSpace ? "SPACE" : letter || "READY"}</strong>
      </div>

      <svg viewBox="0 0 520 430" role="img" aria-label={isSpace ? "Pause between words" : "Animated fingerspelling pose for " + (letter || "no letter")}>
        <defs>
          <linearGradient id="avatar-shirt" x1="0" x2="1">
            <stop offset="0" stopColor={preferences.shirtColor} />
            <stop offset="1" stopColor={preferences.shirtColor} stopOpacity=".72" />
          </linearGradient>
        </defs>

        <circle cx="260" cy="208" r="180" fill="var(--avatar-glow)" />
        <path d="M135 430c5-88 44-134 125-134s120 46 125 134" fill="url(#avatar-shirt)" />
        <path d="M202 306h116v92H202z" fill={preferences.skinTone} opacity=".95" />

        <g className="avatar-head">
          <circle cx="260" cy="177" r="72" fill={preferences.skinTone} />
          <path d="M191 174c0-62 28-92 69-92 54 0 72 37 69 90-18-21-38-31-60-31-31 0-54 14-78 33z" fill={preferences.hairColor} />
          {preferences.hair === "curly" && (
            <>
              <circle cx="202" cy="128" r="14" fill={preferences.hairColor} />
              <circle cx="224" cy="106" r="16" fill={preferences.hairColor} />
              <circle cx="250" cy="99" r="17" fill={preferences.hairColor} />
              <circle cx="278" cy="105" r="16" fill={preferences.hairColor} />
              <circle cx="306" cy="125" r="14" fill={preferences.hairColor} />
            </>
          )}
          {preferences.hair === "long" && (
            <>
              <path d="M193 155c-10 58 7 92 30 108h18V170z" fill={preferences.hairColor} />
              <path d="M327 155c10 58-7 92-30 108h-18V170z" fill={preferences.hairColor} />
            </>
          )}
          <circle cx="235" cy="182" r="4" fill="#111" />
          <circle cx="285" cy="182" r="4" fill="#111" />
          <path d="M242 213c12 8 24 8 36 0" fill="none" stroke="#111" strokeWidth="4" strokeLinecap="round" />
        </g>

        <g className="avatar-arm avatar-arm-left" style={{ transform: "rotate(" + pose.left + "deg) translateY(" + pose.lift + "px)", transformOrigin: "202px 320px" }}>
          <path d="M205 320c-35 25-55 49-69 86" fill="none" stroke={preferences.skinTone} strokeWidth="28" strokeLinecap="round" />
          <g className="avatar-hand" style={{ transform: "translateY(" + (-pose.spread) + "px)" }}>
            <circle cx="136" cy="399" r="20" fill={preferences.skinTone} />
            <path d="M136 390c-20-17-33-25-40-18 0 7 9 15 25 25M137 390c-5-23-5-35 4-36 8 3 10 15 8 34M145 392c8-21 15-31 23-27 6 5 0 17-11 32M148 398c15-11 28-14 33-7 2 8-11 13-29 17" fill="none" stroke={preferences.skinTone} strokeWidth="10" strokeLinecap="round" />
          </g>
        </g>

        <g className="avatar-arm avatar-arm-right" style={{ transform: "rotate(" + pose.right + "deg) translateY(" + pose.lift + "px)", transformOrigin: "318px 320px" }}>
          <path d="M315 320c35 25 55 49 69 86" fill="none" stroke={preferences.skinTone} strokeWidth="28" strokeLinecap="round" />
          <g className="avatar-hand" style={{ transform: "translateY(" + pose.spread + "px)" }}>
            <circle cx="384" cy="399" r="20" fill={preferences.skinTone} />
            <path d="M384 390c20-17 33-25 40-18 0 7-9 15-25 25M383 390c5-23 5-35-4-36-8 3-10 15-8 34M375 392c-8-21-15-31-23-27-6 5 0 17 11 32M372 398c-15-11-28-14-33-7-2 8 11 13 29 17" fill="none" stroke={preferences.skinTone} strokeWidth="10" strokeLinecap="round" />
          </g>
        </g>
      </svg>
    </div>
  );
}
