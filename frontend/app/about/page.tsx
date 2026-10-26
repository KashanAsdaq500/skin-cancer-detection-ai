import Link from "next/link";
import Image from "next/image";
import { 
  ShieldAlert, 
  Info, 
  Code2, 
  Sparkles, 
  FileText, 
  ExternalLink,
  HeartPulse,
  Scale
} from "lucide-react";

function GithubIcon({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

export default function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
      {/* Page Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FDF2F4] border border-[#F8CCD5] text-[#8A1538] text-xs font-semibold">
          <Info className="w-3.5 h-3.5" />
          <span>Institutional Research Mission</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
          Research Mission & Ethical Disclaimers
        </h1>
        <p className="text-xs sm:text-sm text-gray-600 max-w-3xl leading-relaxed">
          Understanding the academic purpose, scientific boundaries, and software architecture behind the Dermora AI project.
        </p>
      </div>

      {/* Brand Identity & Mission Card */}
      <div className="p-6 sm:p-8 rounded-xl bg-[#F8F9FA] border border-gray-200 grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
        <div className="md:col-span-4 flex justify-center">
          <div className="relative w-48 h-32 rounded-lg overflow-hidden border border-gray-200 bg-white p-2 shadow-xs">
            <Image
              src="/dermora-logo.jpg"
              alt="Dermora AI Brand Identity"
              width={1408}
              height={768}
              className="w-full h-full object-contain"
            />
          </div>
        </div>
        <div className="md:col-span-8 space-y-2.5">
          <h2 className="text-lg font-bold text-gray-900">The Dermora AI Vision</h2>
          <p className="text-xs sm:text-sm text-gray-700 leading-relaxed">
            Skin cancer is among the most prevalent human malignancies globally. While early detection dramatically improves clinical prognosis, access to expert dermatological dermoscopy remains uneven worldwide.
          </p>
          <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
            Dermora AI exists as an educational prototype to study automated visual feature extraction, quality triage, and multi-class probabilistic lesion classification using modern web architecture and computer vision.
          </p>
        </div>
      </div>

      {/* Strict Clinical & Legal Disclaimer */}
      <div className="p-6 sm:p-8 rounded-xl bg-[#FDF2F4] border-2 border-[#F8CCD5] space-y-3.5">
        <div className="flex items-center gap-2.5 text-[#8A1538] font-bold text-base">
          <ShieldAlert className="w-5 h-5 text-[#8A1538] shrink-0" />
          <span>Strict Clinical & Legal Disclaimer</span>
        </div>
        <div className="text-xs sm:text-sm text-gray-700 space-y-3 leading-relaxed">
          <p>
            <strong>1. Not a Medical Diagnostic Device:</strong> Dermora AI is an experimental software demonstration developed for academic, educational, and computational research. It has not undergone clinical trials, has not been certified by the US Food and Drug Administration (FDA), European CE-MDR, or any equivalent regulatory medical authority.
          </p>
          <p>
            <strong>2. No Diagnostic Value:</strong> Algorithmic predictions, confidence scores, and risk classifications produced by this software are mathematical approximations based on image pixel distributions and must NEVER be treated as medical diagnoses, prognostic judgments, or indications for surgical intervention.
          </p>
          <p>
            <strong>3. Seek Professional Care:</strong> If you or anyone you know has a suspicious mole, expanding lesion, irregular pigmentation, itching, or bleeding skin spot, consult a licensed dermatologist or qualified medical physician immediately. Do not delay professional evaluation based on this tool.
          </p>
        </div>
      </div>

      {/* Engineering Stack */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold text-gray-900 tracking-tight">Modernized Engineering Stack</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-5 rounded-lg bg-white border border-gray-200 shadow-2xs space-y-2">
            <div className="flex items-center gap-2 font-bold text-gray-900">
              <Code2 className="w-4 h-4 text-[#8A1538]" />
              <span>Next.js 16 & TypeScript</span>
            </div>
            <p className="text-gray-600 leading-relaxed">
              Replacing the monolithic Streamlit script with a production-grade App Router frontend, strictly typed interfaces, and reactive state management.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-gray-200 shadow-2xs space-y-2">
            <div className="flex items-center gap-2 font-bold text-gray-900">
              <Sparkles className="w-4 h-4 text-[#8A1538]" />
              <span>Clinical Healthcare Design</span>
            </div>
            <p className="text-gray-600 leading-relaxed">
              Clean white and soft grey layouts with institutional burgundy accents inspired by leading university hospital websites.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-gray-200 shadow-2xs space-y-2">
            <div className="flex items-center gap-2 font-bold text-gray-900">
              <HeartPulse className="w-4 h-4 text-[#8A1538]" />
              <span>FastAPI Inference Backend</span>
            </div>
            <p className="text-gray-600 leading-relaxed">
              Asynchronous Python API server powering deterministic image validation, sharpness metrics, and feature scoring without fake random noise.
            </p>
          </div>
        </div>
      </div>

      {/* Repository & Open Source Information */}
      <div className="p-6 rounded-xl bg-[#F8F9FA] border border-gray-200 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="space-y-1 text-center sm:text-left">
          <h3 className="text-gray-900 font-bold text-sm">Source Code & Version Control</h3>
          <p className="text-xs text-gray-600">All original legacy files and commit history have been preserved.</p>
        </div>
        <a
          href="https://github.com/KashanAsdaq500/skin-cancer-detection-ai"
          target="_blank"
          rel="noopener noreferrer"
          className="px-4 py-2 rounded-md bg-white hover:bg-gray-50 text-gray-700 text-xs font-semibold flex items-center gap-2 border border-gray-300 shadow-2xs transition-colors shrink-0"
        >
          <GithubIcon className="w-4 h-4" />
          <span>View on GitHub</span>
          <ExternalLink className="w-3.5 h-3.5 text-gray-400" />
        </a>
      </div>
    </div>
  );
}
