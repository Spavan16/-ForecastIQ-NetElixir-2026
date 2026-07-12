import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ForecastIQ | AI-Powered Revenue Intelligence Platform",
  // BUG fix: "AIgnition 2026" -> "AIgnition 3.0" (official competition name per the Project
  // Brief T&Cs). This is the page's <meta name="description"> tag -- visible in browser tab
  // previews, page source, and any social/link-preview card if the live demo URL is shared.
  description: "From Marketing Spend to Revenue Certainty — Built for NetElixir AIgnition 3.0",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-900 text-slate-50 antialiased selection:bg-[#1F7A78]/30 selection:text-[#2A1F18]">
        {children}
      </body>
    </html>
  );
}
