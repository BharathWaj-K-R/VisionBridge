import { useEffect, useMemo, useState } from "react";
import type { AvatarPreferences } from "./AvatarCustomizer";
import { interpolateAngle, poseFor, type HandFinger } from "./avatarRig";

type Props = {
  letter: string;
  preferences: AvatarPreferences;
  playing: boolean;
  view: "full" | "close";
  expression: "neutral" | "question" | "emphasis";
};

const BODY_SCALE = { slim: 0.88, average: 1, athletic: 1.08 } as const;

function fingerRotation(value: number): string {
  return "rotate(" + value + "deg)";
}

function HandRig({
  side,
  fingers,
  skin,
  outline,
  highContrast,
}: {
  side: "left" | "right";
  fingers: HandFinger;
  skin: string;
  outline: string;
  highContrast: boolean;
}) {
  const mirrored = side === "right";
  const transform = mirrored ? "translate(0 0) scale(-1 1)" : undefined;
  const finger = (x: number, y: number, length: number, rotation: number, key: string) => (
    <g key={key} transform={"translate(" + x + " " + y + ") " + fingerRotation(rotation)} className="finger-joint">
      <line x1="0" y1="0" x2="0" y2={-length * 0.52} stroke={outline} strokeWidth={highContrast ? 3 : 1.5} strokeLinecap="round" opacity={highContrast ? 0.95 : 0.55} />
      <line x1="0" y1="0" x2="0" y2={-length} stroke={skin} strokeWidth="11" strokeLinecap="round" />
      <circle cx="0" cy={-length * 0.52} r="3.1" fill={outline} opacity={highContrast ? 0.85 : 0.35} />
      <circle cx="0" cy={-length} r="2.8" fill={outline} opacity={highContrast ? 0.9 : 0.4} />
    </g>
  );

  return (
    <g transform={transform} className={"hand-rig " + side}>
      <ellipse cx="0" cy="0" rx="34" ry="29" fill={skin} stroke={outline} strokeWidth={highContrast ? 3 : 1.5} />
      {finger(8, -18, 39, fingers.index, "index")}
      {finger(18, -13, 43, fingers.middle, "middle")}
      {finger(27, -7, 39, fingers.ring, "ring")}
      {finger(34, 1, 31, fingers.little, "little")}
      <g transform={"translate(-22 2) " + fingerRotation(fingers.thumb)}>
        <line x1="0" y1="0" x2="-26" y2="-24" stroke={outline} strokeWidth={highContrast ? 3 : 1.5} strokeLinecap="round" opacity={highContrast ? .95 : .55} />
        <line x1="0" y1="0" x2="-28" y2="-26" stroke={skin} strokeWidth="11" strokeLinecap="round" />
      </g>
      {highContrast && <ellipse cx="0" cy="0" rx="38" ry="33" fill="none" stroke="#38d9ff" strokeWidth="2" opacity=".9" />}
    </g>
  );
}

function Apparel({
  kind,
  color,
  bodyScale,
}: {
  kind: AvatarPreferences["apparel"];
  color: string;
  bodyScale: number;
}) {
  const width = 148 * bodyScale;
  if (kind === "vest") {
    return (
      <g className="apparel-layer">
        <path d={"M260 292C177 292 135 340 118 494H402C385 340 343 292 260 292Z"} fill={color} />
        <path d="M207 302L241 352L260 332L279 352L313 302" fill="none" stroke="var(--avatar-trim)" strokeWidth="8" />
      </g>
    );
  }
  if (kind === "button-down") {
    return (
      <g className="apparel-layer">
        <path d={"M260 292C" + (260 - width) + " 296 " + (260 - width - 28) + " 355 112 494H408C408 494 388 355 " + (260 + width + 28) + " 296 260 292Z"} fill={color} />
        <path d="M260 296V494M235 315L260 343L285 315" fill="none" stroke="rgba(255,255,255,.7)" strokeWidth="3" />
        {[350, 384, 418, 452].map((y) => <circle key={y} cx="260" cy={y} r="3.5" fill="rgba(255,255,255,.8)" />)}
      </g>
    );
  }
  return (
    <g className="apparel-layer">
      <path d={"M260 292C177 292 135 340 118 494H402C385 340 343 292 260 292Z"} fill={color} />
      <path d="M212 300C229 317 291 317 308 300" fill="none" stroke="rgba(255,255,255,.35)" strokeWidth="6" />
    </g>
  );
}

function Face({
  skin,
  hair,
  hairColor,
  expression,
}: {
  skin: string;
  hair: AvatarPreferences["hair"];
  hairColor: string;
  expression: Props["expression"];
}) {
  const brows = expression === "question" ? { left: -8, right: 8 } : expression === "emphasis" ? { left: 8, right: -8 } : { left: 0, right: 0 };
  return (
    <g className="face-layer">
      {hair === "long" && <g className="hair-back"><path d="M192 145C180 214 200 245 224 258L238 176Z" fill={hairColor} /><path d="M328 145C340 214 320 245 296 258L282 176Z" fill={hairColor} /></g>}
      <circle cx="260" cy="160" r="72" fill={skin} stroke="var(--avatar-outline)" strokeWidth="2" />
      <path d="M194 155C186 102 212 76 260 76C309 76 336 105 326 156C306 134 286 124 260 124C234 124 213 135 194 155Z" fill={hairColor} className={"hair-front hair-" + hair} />
      {hair === "curly" && <g className="hair-curls">{[198,220,244,270,294,316].map((x, i) => <circle key={i} cx={x} cy={100 + (i % 2) * 7} r="17" fill={hairColor} />)}</g>}
      <g className="expression-layer">
        <path d={"M218 144q18 " + brows.left + " 36 0"} fill="none" stroke="#1a1715" strokeWidth="5" strokeLinecap="round" />
        <path d={"M266 144q18 " + brows.right + " 36 0"} fill="none" stroke="#1a1715" strokeWidth="5" strokeLinecap="round" />
        <ellipse cx="236" cy="162" rx="5" ry="7" fill="#171615" />
        <ellipse cx="284" cy="162" rx="5" ry="7" fill="#171615" />
        {expression === "question" ? (
          <path d="M236 199Q260 186 286 200" fill="none" stroke="#1a1715" strokeWidth="4" strokeLinecap="round" />
        ) : expression === "emphasis" ? (
          <path d="M238 198Q260 216 282 198" fill="none" stroke="#1a1715" strokeWidth="4" strokeLinecap="round" />
        ) : (
          <path d="M240 199Q260 212 280 199" fill="none" stroke="#1a1715" strokeWidth="4" strokeLinecap="round" />
        )}
      </g>
    </g>
  );
}

export default function SignAvatar({ letter, preferences, playing, view, expression }: Props) {
  const pose = useMemo(() => poseFor(letter), [letter]);
  const [motionT, setMotionT] = useState(1);
  const previousLetter = useMemo(() => letter || "", [letter]);

  useEffect(() => {
    let frame = 0;
    const start = performance.now();
    const duration = playing ? 280 : 180;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setMotionT(t);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [previousLetter, playing]);

  const leftArm = interpolateAngle(0, pose.leftArm, motionT);
  const rightArm = interpolateAngle(0, pose.rightArm, motionT);
  const leftElbow = interpolateAngle(0, pose.leftElbow, motionT);
  const rightElbow = interpolateAngle(0, pose.rightElbow, motionT);
  const outline = preferences.highContrast ? "#38d9ff" : "#3a302b";
  const bodyScale = BODY_SCALE[preferences.bodyShape];
  const crop = view === "close" ? "76 55 368 390" : "0 0 520 520";
  const displaySign = letter === " " ? "SPACE" : letter || "READY";
  const stageClass = preferences.highContrast ? "sign-avatar high-contrast" : "sign-avatar";

  return (
    <div className={playing ? stageClass + " playing" : stageClass} data-letter={letter || ""}>
      <div className="avatar-stage-label">
        <div><span className="eyebrow">CURRENT SIGN</span><strong>{displaySign}</strong></div>
        <span className="avatar-expression-tag">{expression.toUpperCase()}</span>
      </div>

      <svg viewBox={crop} role="img" aria-label={"2D signing avatar, current sign " + (displaySign === "SPACE" ? "space" : displaySign.toLowerCase())}>
        <defs>
          <linearGradient id="avatar-shirt-refined" x1="0" x2="1">
            <stop offset="0" stopColor={preferences.shirtColor} />
            <stop offset="1" stopColor={preferences.shirtColor} stopOpacity=".72" />
          </linearGradient>
          <radialGradient id="avatar-stage-refined">
            <stop offset="0" stopColor="var(--avatar-glow)" />
            <stop offset="1" stopColor="var(--avatar-stage-bg)" />
          </radialGradient>
        </defs>

        <rect x="0" y="0" width="520" height="520" fill="url(#avatar-stage-refined)" />
        <circle cx="260" cy="210" r="190" fill="var(--avatar-glow)" />

        <g className={playing ? "avatar-motion-sign" : "avatar-motion-idle"}>
          <g className="avatar-root" style={{ transform: "translateX(" + (260 * (1 - bodyScale)) + "px) scaleX(" + bodyScale + ")" }}>
          <Apparel kind={preferences.apparel} color={preferences.shirtColor} bodyScale={bodyScale} />

          <path d="M210 286L184 308L154 370" fill="none" stroke={preferences.skinTone} strokeWidth="31" strokeLinecap="round" />
          <path d="M310 286L336 308L366 370" fill="none" stroke={preferences.skinTone} strokeWidth="31" strokeLinecap="round" />

          <g className="bone-arm" style={{ transform: "rotate(" + leftArm + "deg)", transformOrigin: "184px 308px" }}>
            <path d="M184 308L146 358" fill="none" stroke="var(--avatar-sleeve)" strokeWidth="34" strokeLinecap="round" opacity=".84" />
            <g style={{ transform: "rotate(" + leftElbow + "deg)", transformOrigin: "146px 358px" }}>
              <path d="M146 358L116 223" fill="none" stroke={preferences.skinTone} strokeWidth="25" strokeLinecap="round" />
              <g transform={"translate(116 222) rotate(" + (pose.leftWrist - 12) + ")"}>
                <HandRig side="left" fingers={pose.leftHand} skin={preferences.skinTone} outline={outline} highContrast={preferences.highContrast} />
              </g>
            </g>
          </g>

          <g className="bone-arm" style={{ transform: "rotate(" + rightArm + "deg)", transformOrigin: "336px 308px" }}>
            <path d="M336 308L374 358" fill="none" stroke="var(--avatar-sleeve)" strokeWidth="34" strokeLinecap="round" opacity=".84" />
            <g style={{ transform: "rotate(" + rightElbow + "deg)", transformOrigin: "374px 358px" }}>
              <path d="M374 358L404 223" fill="none" stroke={preferences.skinTone} strokeWidth="25" strokeLinecap="round" />
              <g transform={"translate(404 222) rotate(" + (pose.rightWrist + 12) + ")"}>
                <HandRig side="right" fingers={pose.rightHand} skin={preferences.skinTone} outline={outline} highContrast={preferences.highContrast} />
              </g>
            </g>
          </g>

          <path d="M205 294C214 281 232 274 260 274C288 274 306 281 315 294L310 316C286 325 234 325 210 316Z" fill={preferences.skinTone} opacity=".96" />
          <Face skin={preferences.skinTone} hair={preferences.hair} hairColor={preferences.hairColor} expression={expression} />
          </g>
        </g>

        {preferences.highContrast && (
          <g className="contrast-guides">
            <line x1="78" y1="448" x2="442" y2="448" stroke="#38d9ff" strokeWidth="2" opacity=".25" />
            <circle cx="116" cy="223" r="44" fill="none" stroke="#38d9ff" strokeDasharray="5 6" opacity=".35" />
            <circle cx="404" cy="223" r="44" fill="none" stroke="#38d9ff" strokeDasharray="5 6" opacity=".35" />
          </g>
        )}
      </svg>

      <div className="avatar-status-strip">
        <span>{playing ? "LISTENING / SIGNING" : "READY FOR NEXT SIGN"}</span>
        <div className="avatar-meter" aria-hidden="true">{[0,1,2,3,4].map((bar) => <i key={bar} className={playing ? "on" : ""} />)}</div>
      </div>
    </div>
  );
}
