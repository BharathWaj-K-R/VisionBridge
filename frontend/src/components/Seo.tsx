import { useEffect } from "react";
import { useLocation } from "react-router-dom";

const SITE = "https://visionbridge-2c7h.onrender.com";
const META: Record<string, { title: string; description: string }> = {
  "/": { title: "VisionBridge | ISL A–Z recognition", description: "VisionBridge recognizes Indian Sign Language A–Z letters from browser camera input and supports signer-specific calibration." },
  "/login": { title: "Sign in | VisionBridge", description: "Sign in to VisionBridge to use camera-based ISL A–Z recognition, signer calibration, and recognition history." },
  "/dashboard": { title: "Home | VisionBridge", description: "Choose camera translation or speech input, then access saved communication phrases and recent recognition." },
  "/translate": { title: "Camera translation | VisionBridge", description: "Use browser camera input for real-time Indian Sign Language A–Z recognition with optional signer calibration." },
  "/calibration": { title: "Personalize recognition | VisionBridge", description: "Capture three examples per selected letter to create a signer-specific recognition adapter." },
  "/history": { title: "History | VisionBridge", description: "Review and export recorded Indian Sign Language letter recognition events." },
  "/settings": { title: "Settings | VisionBridge", description: "Manage account access, appearance, camera behavior, browser permissions, and signer adapters." },
  "/voice-to-sign": { title: "Speak | VisionBridge", description: "Speak a sentence and review the available sign references in sequence." },
  "/word-bank": { title: "Phrases & Quick Access | VisionBridge", description: "Browse communication phrases, add custom phrases, and manage ten Quick Access slots." },
  "/personalization": { title: "My profile | VisionBridge", description: "Manage profiles for avatar appearance, Quick Access phrases, favorites, signing speed, and speech voice." },
  "/404": { title: "Page not found | VisionBridge", description: "The requested VisionBridge page does not exist. Return to Home or choose another application section." },
};

function setMeta(name: string, content: string) {
  let node = document.head.querySelector<HTMLMetaElement>(`meta[name="${name}"]`);
  if (!node) { node = document.createElement("meta"); node.name = name; document.head.appendChild(node); }
  node.content = content;
}
function setProperty(property: string, content: string) {
  let node = document.head.querySelector<HTMLMetaElement>(`meta[property="${property}"]`);
  if (!node) { node = document.createElement("meta"); node.setAttribute("property", property); document.head.appendChild(node); }
  node.content = content;
}
function setLink(rel: string, href: string) {
  let node = document.head.querySelector<HTMLLinkElement>(`link[rel="${rel}"]`);
  if (!node) { node = document.createElement("link"); node.rel = rel; document.head.appendChild(node); }
  node.href = href;
}
export default function Seo() {
  const { pathname } = useLocation();
  useEffect(() => {
    const meta = META[pathname] || META["/404"];
    const canonical = pathname === "/" ? SITE + "/" : SITE + pathname;
    document.title = meta.title;
    setMeta("description", meta.description);
    setMeta("robots", pathname === "/" || pathname === "/login" ? "index,follow" : "noindex,nofollow");
    setProperty("og:title", meta.title); setProperty("og:description", meta.description);
    setProperty("og:type", "website"); setProperty("og:url", canonical);
    setProperty("og:image", SITE + "/social-share.svg"); setProperty("og:image:secure_url", SITE + "/social-share.svg"); setProperty("og:image:type", "image/svg+xml"); setProperty("og:image:width", "1200"); setProperty("og:image:height", "630"); setProperty("og:image:alt", "VisionBridge Indian Sign Language recognition social preview");
    setProperty("og:site_name", "VisionBridge");
    setMeta("twitter:card", "summary_large_image"); setMeta("twitter:title", meta.title);
    setMeta("twitter:description", meta.description); setMeta("twitter:image", SITE + "/social-share.svg"); setMeta("twitter:image:alt", "VisionBridge Indian Sign Language recognition social preview");
    setLink("canonical", canonical);
    document.getElementById("visionbridge-structured-data")?.remove();
    const script = document.createElement("script");
    script.id = "visionbridge-structured-data"; script.type = "application/ld+json";
    const name = META[pathname]?.title.replace(" | VisionBridge", "") || "Page";
    const breadcrumbItems = pathname === "/" ? [{ "@type":"ListItem", position:1, name:"VisionBridge", item:SITE+"/" }] : [
      { "@type":"ListItem", position:1, name:"Dashboard", item:SITE+"/dashboard" },
      { "@type":"ListItem", position:2, name, item:canonical },
    ];
    script.textContent = JSON.stringify([
      { "@context":"https://schema.org", "@type":"SoftwareApplication", name:"VisionBridge", applicationCategory:"EducationalApplication", operatingSystem:"Web", url:SITE+"/", description:META["/"].description },
      { "@context":"https://schema.org", "@type":"BreadcrumbList", itemListElement:breadcrumbItems },
    ]);
    document.head.appendChild(script);
    return () => script.remove();
  }, [pathname]);
  return null;
}