"use client";

import { useEffect, useState, Suspense } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getApiUrl } from "@/lib/api";
import { 
  ShieldAlert, 
  ArrowLeft, 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  Info, 
  Layers, 
  Scan,
  Cpu,
  BarChart3,
  FileText,
  Printer,
  XCircle,
  HelpCircle,
  CheckCircle
} from "lucide-react";

interface ClassProb {
  class_name: string;
  code: string;
  probability: number;
  risk_level: string;
  clinical_description: string;
}

interface ImageMetrics {
  width: number;
  height: number;
  aspect_ratio: number;
  sharpness_score: number;
  contrast_score: number;
  mean_luminance: number;
  skin_tone_ratio: number;
  is_supported_dermoscopic_input: boolean;
  quality_status: string;
  rejection_reasons: string[];
  quality_notes: string[];
}

interface PredictionData {
  id: string;
  timestamp: string;
  is_genuine_model_active: boolean;
  analysis_status: string;
  triage_message: string;
  is_supported_dermoscopic_input: boolean;
  rejection_reasons: string[];
  predicted_class?: string | null;
  predicted_code?: string | null;
  confidence?: number | null;
  risk_level?: string | null;
  probabilities: ClassProb[];
  image_metrics: ImageMetrics;
  model_version: string;
  model_status: string;
  is_verified_clinical_model: boolean;
  training_dataset_target: string;
  disclaimer: string;
  limitations: string[];
  image_preview?: string;
}

function ResultDetailContent() {
  const params = useParams();
  const router = useRouter();
  const resultId = params?.id as string;

  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!resultId) return;

    async function loadResult() {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(getApiUrl(`/api/results/${resultId}`));
        if (res.ok) {
          const json = await res.json();
          setData(json);
          setLoading(false);
          return;
        }
      } catch (err) {
        console.warn("Backend fetch failed, attempting sessionStorage fallback:", err);
      }

      if (typeof window !== "undefined") {
        const cached = sessionStorage.getItem(`dermora_result_${resultId}`);
        if (cached) {
          try {
            setData(JSON.parse(cached));
            setLoading(false);
            return;
          } catch (e) {
            console.error("Failed to parse cached result", e);
          }
        }
      }

      setError("Analysis report not found. The record may have expired or the backend server is unreachable.");
      setLoading(false);
    }

    loadResult();
  }, [resultId]);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-24 text-center space-y-4">
        <div className="w-10 h-10 border-4 border-gray-200 border-t-[#8A1538] rounded-full animate-spin mx-auto" />
        <h2 className="text-lg font-bold text-gray-900">Retrieving Clinical Analysis Report...</h2>
        <p className="text-xs text-gray-500">Validating input domain and quality metrics</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-20 text-center space-y-6">
        <div className="w-14 h-14 rounded-full bg-red-50 border border-red-200 text-red-700 flex items-center justify-center mx-auto">
          <AlertTriangle className="w-7 h-7" />
        </div>
        <div className="space-y-1.5">
          <h2 className="text-2xl font-bold text-gray-900">Report Unavailable</h2>
          <p className="text-xs sm:text-sm text-gray-600">{error || "Could not retrieve analysis data."}</p>
        </div>
        <Link
          href="/scan"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-[#8A1538] text-white font-semibold text-xs sm:text-sm hover:bg-[#70102D] transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Return to Scanner</span>
        </Link>
      </div>
    );
  }

  const isInputRejected = !data.is_supported_dermoscopic_input;
  const isGenuineActive = data.is_genuine_model_active;

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Top Clinical Header & Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-gray-200 pb-4">
        <Link
          href="/scan"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-gray-600 hover:text-[#8A1538] transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Perform Another Lesion Scan</span>
        </Link>
        <div className="flex items-center gap-3 text-xs text-gray-500 font-mono">
          <span>REPORT ID:</span>
          <span className="text-gray-900 px-2 py-0.5 rounded bg-gray-100 border border-gray-200 font-bold">{data.id.slice(0, 8)}</span>
          <button 
            type="button" 
            onClick={() => window.print()} 
            className="hidden sm:inline-flex items-center gap-1 px-2.5 py-1 rounded bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 text-xs font-sans font-medium transition-colors cursor-pointer"
          >
            <Printer className="w-3.5 h-3.5 text-gray-500" />
            <span>Print Report</span>
          </button>
        </div>
      </div>

      {/* CASE 1: INPUT REJECTED (Non-skin image, blur, extreme lighting, low resolution) */}
      {isInputRejected && (
        <div className="p-6 sm:p-8 rounded-xl bg-red-50/70 border border-red-300 space-y-5">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-full bg-red-100 text-red-700 flex items-center justify-center shrink-0 mt-0.5">
              <XCircle className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-red-800">Quality Triage Filter</span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-red-100 text-red-800 border border-red-200">
                  Input Rejected
                </span>
              </div>
              <h2 className="text-xl font-bold text-gray-900">Photograph Unsuitable for Lesion Analysis</h2>
              <p className="text-xs sm:text-sm text-gray-700 leading-relaxed">
                {data.triage_message}
              </p>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-white border border-red-200 space-y-2">
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">Identified Quality / Domain Failures:</h3>
            <ul className="space-y-1.5 text-xs text-red-800">
              {data.rejection_reasons.map((reason, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="font-bold text-red-600">&times;</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-white/80 border border-gray-200 text-xs text-gray-600 space-y-1.5">
            <div className="font-semibold text-gray-800 flex items-center gap-1.5">
              <HelpCircle className="w-4 h-4 text-[#8A1538]" />
              <span>Dermatological Capture Requirements:</span>
            </div>
            <p className="leading-relaxed">
              Dermora AI is exclusively trained for dermoscopic and close-up macroscopic skin lesion photography. Arbitrary internet images, selfies, text documents, or non-cutaneous objects are automatically rejected to maintain clinical integrity.
            </p>
          </div>

          <div className="pt-2">
            <Link
              href="/scan"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-[#8A1538] hover:bg-[#70102D] text-white text-xs sm:text-sm font-semibold transition-colors"
            >
              <Scan className="w-4 h-4" />
              <span>Upload Supported Lesion Photograph</span>
            </Link>
          </div>
        </div>
      )}

      {/* Main Two-Column Report Layout (Shown when input is supported) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Image Preview & Technical Metrics */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-5 rounded-xl bg-white border border-gray-200 shadow-2xs space-y-3">
            <div className="flex items-center justify-between text-xs text-gray-600 border-b border-gray-100 pb-2">
              <span className="font-semibold text-gray-900">Inspected Photograph</span>
              <span>{data.image_metrics.width} × {data.image_metrics.height} px</span>
            </div>

            {data.image_preview ? (
              <div className="relative rounded-lg overflow-hidden border border-gray-200 bg-gray-50 flex items-center justify-center max-h-80 p-1">
                <img
                  src={data.image_preview}
                  alt="Dermoscopic Lesion"
                  className="w-full h-auto max-h-72 object-contain mx-auto rounded"
                />
              </div>
            ) : (
              <div className="h-64 rounded-lg bg-gray-50 border border-gray-200 flex items-center justify-center text-xs text-gray-400">
                Image preview unavailable
              </div>
            )}

            {/* Quality Metrics Grid */}
            <div className="grid grid-cols-2 gap-2.5 pt-1">
              <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
                <div className="text-[10px] text-gray-500 uppercase font-semibold">Sharpness Index</div>
                <div className="text-lg font-bold text-gray-900">{data.image_metrics.sharpness_score} <span className="text-xs font-normal text-gray-500">/ 100</span></div>
              </div>
              <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
                <div className="text-[10px] text-gray-500 uppercase font-semibold">Contrast Score</div>
                <div className="text-lg font-bold text-gray-900">{data.image_metrics.contrast_score} <span className="text-xs font-normal text-gray-500">/ 100</span></div>
              </div>
            </div>

            {data.image_metrics.quality_notes && data.image_metrics.quality_notes.length > 0 && (
              <div className="p-3 rounded-lg bg-[#F8F9FA] border border-gray-200 text-[11px] text-gray-600 space-y-1">
                <div className="font-semibold text-gray-800">Image Triage Verification:</div>
                {data.image_metrics.quality_notes.map((note, idx) => (
                  <div key={idx} className="flex items-center gap-1.5 text-gray-600">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                    <span>{note}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Model Status Card */}
          <div className="p-4 rounded-xl bg-[#F8F9FA] border border-gray-200 text-xs space-y-1.5">
            <div className="flex items-center gap-2 text-gray-900 font-semibold">
              <Cpu className="w-4 h-4 text-[#8A1538]" />
              <span>Inference Engine Status</span>
            </div>
            <div className="text-gray-800 font-mono text-[11px] font-medium">{data.model_version}</div>
            <div className="text-gray-600 text-[11px] leading-relaxed">{data.model_status}</div>
          </div>
        </div>

        {/* Right Column: Model Status / Probabilities / Limitations */}
        <div className="lg:col-span-7 space-y-6">
          {/* CASE 2A: Supported input, but Genuine Weights Pending */}
          {!isInputRejected && !isGenuineActive && (
            <div className="p-6 rounded-xl bg-white border border-gray-200 shadow-2xs space-y-4">
              <div className="flex items-center justify-between border-b border-gray-100 pb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                  Dermoscopic Triage Assessment
                </span>
                <span className="text-xs font-bold px-3 py-1 rounded-full border bg-blue-50 text-blue-800 border-blue-200">
                  Triage Passed
                </span>
              </div>

              <div className="space-y-2">
                <h3 className="text-xl font-bold text-gray-900">
                  Photograph Validated &bull; Neural Network Training Pending
                </h3>
                <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                  {data.triage_message}
                </p>
              </div>

              <div className="p-4 rounded-lg bg-[#F8F9FA] border border-gray-200 space-y-2 text-xs">
                <div className="font-bold text-gray-900 flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4 text-[#8A1538]" />
                  <span>Zero Synthetic / Fake Predictions Policy:</span>
                </div>
                <p className="text-gray-600 leading-relaxed">
                  In strict accordance with medical AI safety guidelines, Dermora AI does not display pseudo-probabilities, simulated risk scores, or ungrounded cancer predictions. True 7-class lesion probability vectors will only be activated once model training on the official HAM10000 dataset is executed.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-emerald-50/60 border border-emerald-200 text-xs space-y-1.5">
                <div className="font-bold text-emerald-900">Next Step for Model Activation:</div>
                <p className="text-emerald-800 leading-relaxed">
                  Follow the training guide in <code className="bg-white px-1.5 py-0.5 rounded border border-emerald-300 font-mono text-[11px]">backend/app/training/DATASET_SETUP.md</code> to acquire the CC BY-NC 4.0 HAM10000 archive and run the reproducible PyTorch training pipeline.
                </p>
              </div>
            </div>
          )}

          {/* CASE 2B: Genuine Deep Learning Weights Active */}
          {!isInputRejected && isGenuineActive && (
            <>
              <div className="p-6 rounded-xl bg-white border border-gray-200 shadow-2xs space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                    Neural Network Prediction
                  </span>
                  <span className="text-xs font-bold px-3 py-1 rounded-full border bg-red-50 text-red-800 border-red-300">
                    {data.risk_level}
                  </span>
                </div>

                <div className="space-y-1">
                  <div className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight flex items-center gap-3">
                    <span>{data.predicted_class}</span>
                    <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-gray-100 text-gray-700 border border-gray-200">
                      {data.predicted_code}
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 leading-relaxed">
                    Genuine model prediction with <strong>{data.confidence}%</strong> relative softmax probability score. Note: This is an uncalibrated algorithmic output, not clinical diagnostic certainty, and cannot rule out malignancy.
                  </p>
                </div>
              </div>

              {/* Critical False-Negative & Melanoma Sensitivity Warning */}
              <div className="p-5 rounded-xl bg-amber-50/80 border-2 border-amber-300 text-amber-950 space-y-2.5">
                <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-amber-900">
                  <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
                  <span>Clinical Safety Notice: Benign Prediction Does Not Rule Out Skin Cancer</span>
                </div>
                <p className="text-xs text-amber-900 leading-relaxed">
                  In benchmark independent evaluation on 1,512 unseen HAM10000 lesions, the model achieved an overall balanced accuracy of <strong>70.4%</strong>, but an independent <strong>Melanoma sensitivity of only 45.3%</strong>: 73 of 161 melanomas were detected; 88 were missed or misclassified. Of these, 46 were predicted as melanocytic nevi and 22 as benign keratoses; the remaining 20 were classified into other categories.
                </p>
                <p className="text-xs text-amber-950 font-semibold leading-relaxed">
                  A benign prediction (such as Melanocytic Nevus) must NEVER be used to dismiss a lesion. If you observe any lesion that is changing, irregular in shape/color, bleeding, itching, or growing, seek professional dermatological assessment immediately.
                </p>
              </div>

              {/* Class Probability Breakdown */}
              <div className="p-6 rounded-xl bg-white border border-gray-200 shadow-2xs space-y-4">
                <div className="flex items-center justify-between border-b border-gray-100 pb-3">
                  <div className="flex items-center gap-2 text-sm font-bold text-gray-900">
                    <BarChart3 className="w-4 h-4 text-[#8A1538]" />
                    <span>ISIC 7-Class Probability Distribution</span>
                  </div>
                  <span className="text-[11px] text-gray-500">Trained on HAM10000</span>
                </div>

                <div className="space-y-3.5 pt-1">
                  {data.probabilities.map((item) => (
                    <div key={item.code} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-gray-500 w-12">{item.code}</span>
                          <span className="font-semibold text-gray-800">{item.class_name}</span>
                        </div>
                        <span className="font-mono font-bold text-gray-900">{item.probability}%</span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-gray-100 overflow-hidden border border-gray-200">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            item.risk_level === "High Risk"
                              ? "bg-[#8A1538]"
                              : item.risk_level === "Medium Risk"
                              ? "bg-amber-600"
                              : "bg-emerald-600"
                          }`}
                          style={{ width: `${Math.max(item.probability, 2)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Clinical Limitations Box */}
          <div className="p-6 rounded-xl bg-[#FDF2F4] border border-[#F8CCD5] space-y-2.5">
            <div className="flex items-center gap-2 text-[#8A1538] font-bold text-sm">
              <ShieldAlert className="w-4 h-4 text-[#8A1538] shrink-0" />
              <span>Mandatory Clinical Research Limitations</span>
            </div>
            <ul className="space-y-1.5 text-xs text-gray-700 leading-relaxed">
              {data.limitations.map((limit, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-[#8A1538] font-bold">&bull;</span>
                  <span>{limit}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Action Navigation */}
          <div className="flex flex-col sm:flex-row gap-3 pt-2">
            <Link
              href="/scan"
              className="flex-1 py-3 px-4 rounded-md bg-[#8A1538] hover:bg-[#70102D] text-white font-semibold text-xs sm:text-sm text-center flex items-center justify-center gap-2 transition-colors shadow-xs"
            >
              <Scan className="w-4 h-4" />
              <span>Scan Another Photograph</span>
            </Link>
            <Link
              href="/insights"
              className="flex-1 py-3 px-4 rounded-md bg-white hover:bg-gray-50 border border-gray-300 text-gray-700 font-semibold text-xs sm:text-sm text-center flex items-center justify-center gap-2 transition-colors shadow-xs"
            >
              <Layers className="w-4 h-4 text-gray-500" />
              <span>Dataset Setup & Training Roadmap</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ResultDetailPage() {
  return (
    <Suspense
      fallback={
        <div className="max-w-4xl mx-auto px-4 py-24 text-center space-y-4">
          <div className="w-10 h-10 border-4 border-gray-200 border-t-[#8A1538] rounded-full animate-spin mx-auto" />
          <h2 className="text-lg font-bold text-gray-900">Loading Clinical Report...</h2>
        </div>
      }
    >
      <ResultDetailContent />
    </Suspense>
  );
}
