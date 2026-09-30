import { useEffect } from "react";
import { useLocation } from "react-router-dom";

const SITE = "https://visionbridge-2c7h.onrender.com";
const META: Record<string, { title: string; description: string }> = {
  "/": { title: "VisionBridge | Indian Sign Language Recognition", description: "VisionBridge provides signer-adaptive Indian Sign Language A–Z recognition using hand landmarks in a focused browser-based recognition workspace for ISL." },
  "/login": { title: "Sign In | VisionBridge", description: "Sign in to VisionBridge to access Indian Sign Language letter recognition, signer calibration, recognition history, and signer profile controls in your browser." },
  "/dashboard": { title: "Dashboard | VisionBridge", description: "Review VisionBridge recognition activity, model status, confidence, latency, and direct access to Indian Sign Language recognition tools from one dashboard." },
  "/translate": { title: "Live Translate | VisionBridge", description: "Use VisionBridge Live Translate for real-time Indian Sign Language A–Z recognition with browser camera hand tracking and signer-adaptive inference." },
  "/calibration": { title: "Calibration | VisionBridge", description: "Calibrate signer-specific Indian Sign Language letter prototypes in VisionBridge using a few clean hand examples for personalized recognition." },
  "/history": { title: "Letter History | VisionBridge", description: "Review and export recorded Indian Sign Language letter recognition events from your VisionBridge signer session for quick analysis and reference." },
  "/settings": { title: "Settings | VisionBridge", description: "Manage VisionBridge signer adapters, account preferences, appearance, camera behavior, and secure browser session controls from one settings page." },
  "/voice-to-sign": { title: "Voice to Sign | VisionBridge", description: "Speak a phrase and VisionBridge turns the recognized text into an animated A–Z fingerspelling sequence with a customizable browser avatar." },
  "/word-bank": { title: "Word Bank | VisionBridge", description: "Browse daily communication phrases, speak words aloud, add custom phrases, and configure your ten VisionBridge quick-access slots." },
  "/personalization": { title: "My Profile | VisionBridge", description: "Manage personal VisionBridge profiles for avatar appearance, ten-slot Quick Access, favorites, signing speed, and preferred speech voice." },
  "/404": { title: "Page Not Found | VisionBridge", description: "The requested VisionBridge page could not be found. Return to the dashboard or choose another Indian Sign Language application section today." },
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