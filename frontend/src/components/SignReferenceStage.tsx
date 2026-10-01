import type { SignPlaybackItem } from "../signing/signTypes";

type Props = {
  item: SignPlaybackItem | null;
  sentenceText: string;
  sentenceIndex: number;
  tokenIndex: number;
};

export default function SignReferenceStage({ item, sentenceText, sentenceIndex, tokenIndex }: Props) {
  const asset = item?.asset || null;

  return (
    <div className="sign-reference-stage sentence-reference-stage">
      <div className="sign-reference-corner top-left">
        SENTENCE {sentenceIndex >= 0 ? String(sentenceIndex + 1).padStart(2, "0") : "--"}
      </div>
      <div className="sign-reference-corner top-right">
        {item ? "TOKEN " + String(tokenIndex + 1).padStart(2, "0") : "READY"}
      </div>

      <div className={"sentence-reference-card" + (asset ? " active" : "")}>
        {asset?.backgroundPosition ? (
          <div
            className="sentence-sign-image"
            role="img"
            aria-label={item ? "ISL reference image for " + item.sourceText : "No sign selected"}
            style={{
              backgroundImage: "url(" + asset.src + ")",
              backgroundPosition: asset.backgroundPosition,
              backgroundSize: "700% 400%",
            }}
          />
        ) : asset ? (
          <img className="sentence-sign-image sentence-sign-image-direct" src={asset.src} alt={asset.label} />
        ) : (
          <div className="sentence-sign-placeholder">
            <span className="eyebrow">NO SIGN ASSET</span>
            <strong>{item?.sourceText || "READY"}</strong>
            <small>This token is handled without a visual asset.</small>
          </div>
        )}
        {item && <div className="sentence-reference-token">{item.sourceText}</div>}
      </div>

      <div className="sign-reference-corner bottom-left">
        {sentenceText || "SPEAK TO BUILD A SENTENCE"}
      </div>
      <div className="sign-reference-corner bottom-right">
        {item?.kind?.toUpperCase() || "IDLE"}
      </div>
      <div className="sign-reference-crosshair" />
    </div>
  );
}
