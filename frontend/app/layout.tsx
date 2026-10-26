import type { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import "./globals.css";
import { ShieldAlert, Activity, Database, Info, UploadCloud, ArrowRight } from "lucide-react";

export const metadata: Metadata = {
  title: "Dermora AI - Clinical Dermatology Computer Vision Prototype",
  description: "Standardized dermatological screening and skin lesion classification research prototype powered by computer vision.",
  icons: [
    {
      rel: "icon",
      url: "/dermora-logo.jpg",
      type: "image/jpeg",
    },
  ],
};

function GithubIcon({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white text-gray-800 flex flex-col font-sans antialiased selection:bg-[#8A1538]/10 selection:text-[#8A1538]">
        {/* AKUH-Inspired Healthcare Advisory Banner */}
        <div className="bg-[#FDF2F4] border-b border-[#F8CCD5] text-[#8A1538] text-xs sm:text-sm py-2 px-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-[#8A1538] shrink-0" />
              <span>
                <strong className="font-semibold text-[#8A1538]">Research Prototype Notice:</strong> Dermora AI is an educational computer-vision research platform. It is not an FDA/CE cleared diagnostic service. Always consult a qualified dermatologist for clinical evaluation.
              </span>
            </div>
            <Link
              href="/about"
              className="hidden md:inline-flex items-center font-semibold text-[#8A1538] hover:text-[#70102D] underline shrink-0"
            >
              Clinical Disclaimers &rarr;
            </Link>
          </div>
        </div>

        {/* Clean Clinical Navbar */}
        <header className="sticky top-0 z-40 bg-white border-b border-gray-200 shadow-xs">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between gap-4">
            {/* Logo & Brand Identity */}
            <Link href="/" className="flex items-center gap-3.5 group focus:outline-none focus:ring-2 focus:ring-[#8A1538] rounded-md p-1">
              <div className="relative w-14 h-14 sm:w-16 sm:h-16 rounded-lg overflow-hidden border border-gray-200 bg-white p-1 shadow-xs flex items-center justify-center">
                <Image
                  src="/dermora-logo.jpg"
                  alt="Dermora AI Logo"
                  width={1408}
                  height={768}
                  className="w-full h-full object-contain"
                  priority
                />
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-lg sm:text-xl text-gray-900 tracking-tight">
                    Dermora<span className="text-[#8A1538]"> AI</span>
                  </span>
                  <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 bg-[#FDF2F4] text-[#8A1538] border border-[#F8CCD5] rounded">
                    Research
                  </span>
                </div>
                <span className="text-xs text-gray-500 hidden sm:inline">
                  Dermatological Screening & Lesion Analysis
                </span>
              </div>
            </Link>

            {/* Navigation Links */}
            <nav className="flex items-center gap-1 sm:gap-2">
              <Link
                href="/"
                className="px-3.5 py-2 text-sm font-medium text-gray-700 hover:text-[#8A1538] hover:bg-gray-50 rounded-md transition-colors"
              >
                Home
              </Link>
              <Link
                href="/scan"
                className="px-3.5 py-2 text-sm font-medium text-gray-700 hover:text-[#8A1538] hover:bg-gray-50 rounded-md transition-colors flex items-center gap-1.5"
              >
                <UploadCloud className="w-4 h-4 text-[#8A1538]" />
                <span>Lesion Scanner</span>
              </Link>
              <Link
                href="/insights"
                className="px-3.5 py-2 text-sm font-medium text-gray-700 hover:text-[#8A1538] hover:bg-gray-50 rounded-md transition-colors flex items-center gap-1.5"
              >
                <Database className="w-4 h-4 text-gray-500" />
                <span className="hidden sm:inline">Model Insights</span>
                <span className="sm:hidden">Insights</span>
              </Link>
              <Link
                href="/about"
                className="px-3.5 py-2 text-sm font-medium text-gray-700 hover:text-[#8A1538] hover:bg-gray-50 rounded-md transition-colors flex items-center gap-1.5"
              >
                <Info className="w-4 h-4 text-gray-500" />
                <span>About</span>
              </Link>
              <Link
                href="/scan"
                className="ml-3 hidden sm:inline-flex items-center gap-2 px-5 py-2.5 bg-[#8A1538] hover:bg-[#70102D] text-white font-semibold text-sm rounded-md shadow-xs transition-colors"
              >
                <Activity className="w-4 h-4" />
                <span>Start Scan</span>
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 bg-white">{children}</main>

        {/* Professional Hospital-Style Footer */}
        <footer className="bg-[#1A202C] text-gray-300 border-t-4 border-[#8A1538] pt-12 pb-8 text-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-4 gap-8">
            {/* Brand Col */}
            <div className="md:col-span-2 space-y-4">
              <div className="flex items-center gap-3">
                <div className="relative w-12 h-12 rounded-md overflow-hidden bg-white p-1 shadow-xs border border-gray-700">
                  <Image
                    src="/dermora-logo.jpg"
                    alt="Dermora AI Logo"
                    width={1408}
                    height={768}
                    className="w-full h-full object-contain"
                  />
                </div>
                <div>
                  <h3 className="text-white font-bold text-lg tracking-tight">Dermora AI</h3>
                  <p className="text-xs text-gray-400">Dermatological Research & Lesion Classification</p>
                </div>
              </div>
              <p className="text-xs text-gray-400 leading-relaxed max-w-md">
                Dermora AI is an educational computer-vision prototype benchmarking multi-class skin lesion image analysis on public datasets including HAM10000 and ISIC archives.
              </p>
              <div className="pt-2">
                <a
                  href="https://github.com/KashanAsdaq500/skin-cancer-detection-ai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-xs text-red-300 hover:text-white font-medium transition-colors"
                >
                  <GithubIcon className="w-4 h-4" />
                  <span>GitHub: KashanAsdaq500/skin-cancer-detection-ai</span>
                </a>
              </div>
            </div>

            {/* Quick Links */}
            <div>
              <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">Healthcare Navigation</h4>
              <ul className="space-y-2 text-xs text-gray-400">
                <li><Link href="/" className="hover:text-white transition-colors">Home Page</Link></li>
                <li><Link href="/scan" className="hover:text-white transition-colors">Lesion Scanner</Link></li>
                <li><Link href="/insights" className="hover:text-white transition-colors">Model Insights & Datasets</Link></li>
                <li><Link href="/about" className="hover:text-white transition-colors">About & Ethics</Link></li>
              </ul>
            </div>

            {/* Clinical Governance */}
            <div>
              <h4 className="text-xs font-semibold text-red-300 uppercase tracking-wider mb-3">Clinical Advisory</h4>
              <p className="text-xs text-gray-400 leading-relaxed">
                This platform is strictly for research demonstration. If you notice a changing, bleeding, or irregular mole, consult a certified dermatologist promptly.
              </p>
            </div>
          </div>

          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-10 pt-6 border-t border-gray-800 text-center text-xs text-gray-500">
            &copy; 2026 Dermora AI. Academic & Research Prototype. All rights reserved.
          </div>
        </footer>
      </body>
    </html>
  );
}
