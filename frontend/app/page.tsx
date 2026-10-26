import Link from "next/link";
import Image from "next/image";
import { 
  ArrowRight, 
  Scan, 
  Layers, 
  AlertTriangle,
  Activity,
  CheckCircle,
  FileText,
  ShieldCheck,
  Stethoscope
} from "lucide-react";

export default function HomePage() {
  const isicClasses = [
    { code: "MEL", name: "Melanoma", type: "Malignant", desc: "Malignant melanocyte neoplasm requiring urgent specialist biopsy.", badge: "bg-red-50 text-red-700 border-red-200" },
    { code: "BCC", name: "Basal Cell Carcinoma", type: "Malignant", desc: "Locally invasive non-melanoma skin cancer from basal epidermis.", badge: "bg-red-50 text-red-700 border-red-200" },
    { code: "AKIEC", name: "Actinic Keratosis", type: "Pre-cancerous", desc: "Sun-damaged intraepithelial dysplasia; warrants medical triage.", badge: "bg-amber-50 text-amber-800 border-amber-200" },
    { code: "NV", name: "Melanocytic Nevus", type: "Benign", desc: "Common benign mole from melanocyte proliferation.", badge: "bg-emerald-50 text-emerald-800 border-emerald-200" },
    { code: "BKL", name: "Benign Keratosis", type: "Benign", desc: "Seborrheic keratosis or solar lentigo; harmless epidermal growth.", badge: "bg-slate-50 text-slate-700 border-slate-200" },
    { code: "DF", name: "Dermatofibroma", type: "Benign", desc: "Benign cutaneous fibrous nodule typically on extremities.", badge: "bg-slate-50 text-slate-700 border-slate-200" },
    { code: "VASC", name: "Vascular Lesion", type: "Benign", desc: "Vascular proliferation (hemangioma, pyogenic granuloma).", badge: "bg-slate-50 text-slate-700 border-slate-200" },
  ];

  const abcdeCriteria = [
    { letter: "A", title: "Asymmetry", desc: "One half of the lesion contour does not mirror the opposite half." },
    { letter: "B", title: "Border", desc: "Edges are ragged, notched, scalloped, blurred, or poorly demarcated." },
    { letter: "C", title: "Color", desc: "Varied pigment distribution featuring shades of tan, brown, black, or blue." },
    { letter: "D", title: "Diameter", desc: "Width exceeds 6mm (approx. pencil eraser), though melanomas can be smaller." },
    { letter: "E", title: "Evolution", desc: "Any noticeable change in lesion size, shape, surface height, or sensation." },
  ];

  return (
    <div className="space-y-16 pb-20">
      {/* Hospital-Style Hero Section */}
      <section className="bg-gradient-to-b from-[#F8F9FA] via-white to-white border-b border-gray-200 pt-12 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left: Headline & Actions */}
          <div className="lg:col-span-7 space-y-6 text-center lg:text-left">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#FDF2F4] border border-[#F8CCD5] text-[#8A1538] text-xs font-semibold">
              <Stethoscope className="w-3.5 h-3.5" />
              <span>Dermatology Computer Vision Platform</span>
            </div>

            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-gray-900 leading-[1.15]">
              Standardized Clinical <br className="hidden sm:inline" />
              <span className="text-[#8A1538]">Skin Lesion Analysis</span>
            </h1>

            <p className="text-base sm:text-lg text-gray-600 leading-relaxed max-w-2xl mx-auto lg:mx-0">
              Dermora AI is a clinical research prototype designed to explore automated dermoscopic feature extraction and multi-class classification across standardized ISIC diagnostic criteria.
            </p>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-3.5 pt-2">
              <Link
                href="/scan"
                className="w-full sm:w-auto px-6 py-3.5 rounded-md bg-[#8A1538] hover:bg-[#70102D] text-white font-semibold text-sm shadow-sm flex items-center justify-center gap-2 transition-colors"
              >
                <Scan className="w-4 h-4" />
                <span>Launch Lesion Scanner</span>
                <ArrowRight className="w-4 h-4 ml-1" />
              </Link>
              <Link
                href="/insights"
                className="w-full sm:w-auto px-6 py-3.5 rounded-md bg-white hover:bg-gray-50 border border-gray-300 text-gray-700 font-semibold text-sm shadow-xs flex items-center justify-center gap-2 transition-colors"
              >
                <Layers className="w-4 h-4 text-gray-500" />
                <span>Model Architecture & Datasets</span>
              </Link>
            </div>

            {/* Institutional Trust Indicators */}
            <div className="grid grid-cols-3 gap-4 pt-6 border-t border-gray-200 max-w-lg mx-auto lg:mx-0">
              <div className="p-3 rounded-lg bg-white border border-gray-200 shadow-2xs">
                <div className="text-xl font-bold text-[#8A1538]">7 Classes</div>
                <div className="text-xs text-gray-500 font-medium">ISIC Taxonomy</div>
              </div>
              <div className="p-3 rounded-lg bg-white border border-gray-200 shadow-2xs">
                <div className="text-xl font-bold text-gray-900">Deterministic</div>
                <div className="text-xs text-gray-500 font-medium">Verified Vision Logic</div>
              </div>
              <div className="p-3 rounded-lg bg-white border border-gray-200 shadow-2xs">
                <div className="text-xl font-bold text-gray-900">Research</div>
                <div className="text-xs text-gray-500 font-medium">Prototype Baseline</div>
              </div>
            </div>
          </div>

          {/* Right: Featured Brand Showcase Card */}
          <div className="lg:col-span-5 flex justify-center">
            <div className="w-full max-w-md rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
              <div className="relative aspect-[16/9] w-full rounded-lg overflow-hidden bg-gray-50 border border-gray-100 flex items-center justify-center p-2">
                <Image
                  src="/dermora-logo.jpg"
                  alt="Dermora AI Brand Identity"
                  width={1408}
                  height={768}
                  className="w-full h-full object-contain"
                  priority
                />
              </div>
              <div className="pt-4 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[#8A1538] uppercase tracking-wider">Clinical Brand Identity</span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-gray-100 text-gray-600 font-medium">Authentic Brandmark</span>
                </div>
                <p className="text-xs text-gray-600 leading-relaxed">
                  The neural network shield emblem and botanical motif symbolize dermatological safeguarding, clinical precision, and patient-centered research.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Clinical Quick Access Cards (AKUH-Inspired Quick Access) */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-gray-900 tracking-tight">Clinical Services & Research Access</h2>
          <p className="text-xs text-gray-500">Explore core functional modules of the Dermora AI prototype</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Link
            href="/scan"
            className="group block p-6 rounded-xl bg-white border border-gray-200 hover:border-[#8A1538] hover:shadow-md transition-all space-y-3"
          >
            <div className="w-10 h-10 rounded-lg bg-[#FDF2F4] text-[#8A1538] flex items-center justify-center group-hover:bg-[#8A1538] group-hover:text-white transition-colors">
              <Scan className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-base text-gray-900 group-hover:text-[#8A1538] transition-colors">
              Dermoscopic Lesion Scanner
            </h3>
            <p className="text-xs text-gray-600 leading-relaxed">
              Upload macroscopic or dermoscopic photographs for image sharpness validation, contrast analysis, and multi-class probabilistic assessment.
            </p>
            <div className="text-xs font-semibold text-[#8A1538] flex items-center gap-1 pt-1">
              <span>Open Scanner &rarr;</span>
            </div>
          </Link>

          <Link
            href="/insights"
            className="group block p-6 rounded-xl bg-white border border-gray-200 hover:border-[#8A1538] hover:shadow-md transition-all space-y-3"
          >
            <div className="w-10 h-10 rounded-lg bg-gray-100 text-gray-700 flex items-center justify-center group-hover:bg-[#8A1538] group-hover:text-white transition-colors">
              <Layers className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-base text-gray-900 group-hover:text-[#8A1538] transition-colors">
              HAM10000 / ISIC Model Insights
            </h3>
            <p className="text-xs text-gray-600 leading-relaxed">
              Review public dataset distributions (10,015 images), balanced multi-class metrics, and the formal PyTorch deep learning fine-tuning roadmap.
            </p>
            <div className="text-xs font-semibold text-[#8A1538] flex items-center gap-1 pt-1">
              <span>View Model Insights &rarr;</span>
            </div>
          </Link>

          <Link
            href="/about"
            className="group block p-6 rounded-xl bg-white border border-gray-200 hover:border-[#8A1538] hover:shadow-md transition-all space-y-3"
          >
            <div className="w-10 h-10 rounded-lg bg-gray-100 text-gray-700 flex items-center justify-center group-hover:bg-[#8A1538] group-hover:text-white transition-colors">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-base text-gray-900 group-hover:text-[#8A1538] transition-colors">
              Ethical AI & Clinical Governance
            </h3>
            <p className="text-xs text-gray-600 leading-relaxed">
              Examine the investigative mission, clinical non-diagnostic disclaimers, biopsy prerequisites, and responsible health technology principles.
            </p>
            <div className="text-xs font-semibold text-[#8A1538] flex items-center gap-1 pt-1">
              <span>Read Governance &rarr;</span>
            </div>
          </Link>
        </div>
      </section>

      {/* ABCDE Clinical Framework Section */}
      <section className="bg-[#F8F9FA] border-y border-gray-200 py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto space-y-10">
          <div className="text-center max-w-2xl mx-auto space-y-2">
            <span className="text-xs font-semibold tracking-wider text-[#8A1538] uppercase">Clinical Guidelines</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">The ABCDE Rule of Lesion Triage</h2>
            <p className="text-xs sm:text-sm text-gray-600">
              Established clinical criteria utilized by dermatologists to distinguish potential melanomas from benign cutaneous nevi.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {abcdeCriteria.map((item) => (
              <div
                key={item.letter}
                className="p-5 rounded-lg bg-white border border-gray-200 shadow-2xs hover:border-[#8A1538] transition-colors flex flex-col justify-between"
              >
                <div>
                  <div className="w-9 h-9 rounded-md bg-[#FDF2F4] border border-[#F8CCD5] text-[#8A1538] font-bold text-lg flex items-center justify-center mb-3">
                    {item.letter}
                  </div>
                  <h3 className="font-bold text-sm text-gray-900 mb-1">{item.title}</h3>
                  <p className="text-xs text-gray-600 leading-relaxed">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ISIC 7-Class Lesion Taxonomy Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto space-y-2 mb-10">
          <span className="text-xs font-semibold tracking-wider text-gray-500 uppercase">Standardized Taxonomy</span>
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">Supported Diagnostic Classes</h2>
          <p className="text-xs sm:text-sm text-gray-600">
            Categorization aligned with the International Skin Imaging Collaboration (ISIC) and the HAM10000 dataset archive.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {isicClasses.map((c) => (
            <div
              key={c.code}
              className="p-5 rounded-lg bg-white border border-gray-200 shadow-2xs hover:border-gray-300 transition-all space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-gray-500">{c.code}</span>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${c.badge}`}>
                  {c.type}
                </span>
              </div>
              <h4 className="font-bold text-gray-900 text-sm">{c.name}</h4>
              <p className="text-xs text-gray-600 leading-relaxed">{c.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Hospital-Grade Advisory Notice */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="p-6 sm:p-8 rounded-xl bg-[#FDF2F4] border border-[#F8CCD5] flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-1.5 text-center md:text-left">
            <div className="flex items-center justify-center md:justify-start gap-2 text-[#8A1538] font-bold text-sm sm:text-base">
              <AlertTriangle className="w-5 h-5 text-[#8A1538]" />
              <span>Academic & Healthcare Research Protocol</span>
            </div>
            <p className="text-xs sm:text-sm text-gray-700 max-w-2xl leading-relaxed">
              Dermora AI is engineered for computational image evaluation and academic dermatological research. It does not replace clinical consultation, dermoscopy by a specialist, or surgical histopathology.
            </p>
          </div>
          <Link
            href="/about"
            className="shrink-0 px-5 py-2.5 rounded-md bg-[#8A1538] hover:bg-[#70102D] text-white text-xs font-semibold shadow-xs transition-colors"
          >
            Review Clinical Disclaimer
          </Link>
        </div>
      </section>
    </div>
  );
}
