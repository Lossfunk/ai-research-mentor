import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Newsreader } from "next/font/google";
import "./globals.css";

const newsreader = Newsreader({ 
  subsets: ["latin"],
  variable: '--font-newsreader',
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Research Canvas",
  description: "Infinite canvas for academic research",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} ${newsreader.variable}`}>
      <body className={GeistSans.className}>{children}</body>
    </html>
  );
}
