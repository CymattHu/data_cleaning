import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SensorSync DataOps Studio",
  description:
    "Web-based multimodal data cleaning, alignment and auto-labeling toolchain for robot skill fine-tuning",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
