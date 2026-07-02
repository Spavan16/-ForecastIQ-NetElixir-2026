import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ForecastIQ | AI-Powered Revenue Intelligence Platform",
  description: "From Marketing Spend to Revenue Certainty — Built for NetElixir AIgnition 2026",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-900 text-slate-50 antialiased selection:bg-sky-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
