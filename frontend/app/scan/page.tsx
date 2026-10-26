"use client";

import { useState, useRef, ChangeEvent, DragEvent } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { getApiUrl, getApiBaseUrl } from "@/lib/api";
import { 
  UploadCloud, 
  Image as ImageIcon, 
  CheckCircle, 
  AlertCircle, 
  ArrowRight, 
  Loader2, 
  ShieldAlert, 
  FileCheck,
  RefreshCw,
  Info
} from "lucide-react";

export default function ScanPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileChange = (file: File) => {
    setErrorMsg(null);
    if (!file.type.startsWith("image/")) {
      setErrorMsg("Please upload a valid image file (JPEG, PNG, or WebP).");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setErrorMsg("File size exceeds 15MB limit. Please upload a smaller image.");
      return;
    }

    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!previewUrl || !selectedFile) {
      setErrorMsg("Please select or drop an image first.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMsg(null);

    try {
      const response = await fetch(getApiUrl("/api/predict"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_base64: previewUrl,
          filename: selectedFile.name,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${response.status}`);
      }

      const resultData = await response.json();
      
      if (typeof window !== "undefined") {
        sessionStorage.setItem(`dermora_result_${resultData.id}`, JSON.stringify(resultData));
      }

      router.push(`/results/${resultData.id}`);
    } catch (err: any) {
      console.error("Analysis error:", err);
      setErrorMsg(
        err.message || 
        `Failed to reach the Dermora AI backend service. Ensure the FastAPI server is running on ${getApiBaseUrl()}.`
      );
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-8">
      {/* Page Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FDF2F4] border border-[#F8CCD5] text-[#8A1538] text-xs font-semibold">
          <UploadCloud className="w-3.5 h-3.5" />
          <span>Lesion Inspection Portal</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
          Skin Lesion Image Inspection
        </h1>
        <p className="text-xs sm:text-sm text-gray-600 max-w-xl mx-auto">
          Submit a high-resolution close-up photograph or dermoscopic capture for computer-vision quality scoring and probabilistic triage.
        </p>
      </div>

      {/* Main Form & Guidelines Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
        {/* Upload Dropzone */}
        <div className="md:col-span-7 space-y-4">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer border-2 border-dashed rounded-xl p-8 text-center transition-all flex flex-col items-center justify-center min-h-[320px] bg-white ${
              isDragging
                ? "border-[#8A1538] bg-[#FDF2F4]/40"
                : "border-gray-300 hover:border-[#8A1538] hover:bg-[#FDF2F4]/15 shadow-2xs"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={onFileInputChange}
            />

            {previewUrl ? (
              <div className="space-y-4 w-full">
                <div className="relative max-h-64 max-w-xs mx-auto rounded-lg overflow-hidden border border-gray-300 shadow-sm bg-gray-50 p-1">
                  <img
                    src={previewUrl}
                    alt="Uploaded lesion preview"
                    className="w-full h-auto max-h-60 object-contain mx-auto"
                  />
                </div>
                <div className="text-xs text-gray-600 flex items-center justify-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-600" />
                  <span className="font-semibold text-gray-800">{selectedFile?.name}</span>
                  <span className="text-gray-500">
                    ({Math.round((selectedFile?.size || 0) / 1024)} KB)
                  </span>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className="inline-flex items-center gap-1.5 text-xs text-[#8A1538] hover:text-[#70102D] font-semibold"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Choose a different photograph</span>
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-14 h-14 rounded-full bg-[#FDF2F4] border border-[#F8CCD5] text-[#8A1538] flex items-center justify-center mx-auto shadow-2xs">
                  <ImageIcon className="w-7 h-7" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-gray-800">
                    Click to select or drag and drop image here
                  </p>
                  <p className="text-xs text-gray-500">
                    Supports high-resolution JPG, PNG, or WebP (max 15MB)
                  </p>
                </div>
              </div>
            )}
          </div>

          {errorMsg && (
            <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-800 text-xs flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="button"
            disabled={!selectedFile || isAnalyzing}
            onClick={handleAnalyze}
            className={`w-full py-3.5 rounded-md font-semibold text-sm flex items-center justify-center gap-2 shadow-xs transition-colors ${
              !selectedFile || isAnalyzing
                ? "bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200"
                : "bg-[#8A1538] hover:bg-[#70102D] text-white cursor-pointer"
            }`}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Processing Computer Vision Metrics...</span>
              </>
            ) : (
              <>
                <FileCheck className="w-4 h-4" />
                <span>Run Dermora AI Analysis</span>
                <ArrowRight className="w-4 h-4 ml-1" />
              </>
            )}
          </button>
        </div>

        {/* Clinical Guidelines Card */}
        <div className="md:col-span-5 space-y-4">
          <div className="p-6 rounded-xl bg-[#F8F9FA] border border-gray-200 space-y-4">
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider flex items-center gap-2">
              <Info className="w-4 h-4 text-[#8A1538]" />
              <span>Clinical Capture Guidelines</span>
            </h3>

            <ul className="space-y-3 text-xs text-gray-600">
              <li className="flex items-start gap-2.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#8A1538] mt-1.5 shrink-0" />
                <span><strong>Center the Lesion:</strong> Ensure the complete border is clearly inside the frame.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#8A1538] mt-1.5 shrink-0" />
                <span><strong>Even Illumination:</strong> Avoid harsh shadows, strong specular glare, or low light.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#8A1538] mt-1.5 shrink-0" />
                <span><strong>Sharp Focal Plane:</strong> Unfocused captures compromise edge gradient and color variance detection.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#8A1538] mt-1.5 shrink-0" />
                <span><strong>Minimize Artifacts:</strong> Move hair aside and avoid surgical marker lines when possible.</span>
              </li>
            </ul>

            <div className="pt-3 border-t border-gray-200">
              <div className="p-3 rounded-md bg-[#FDF2F4] border border-[#F8CCD5] text-[11px] text-[#8A1538] leading-relaxed flex items-start gap-2">
                <ShieldAlert className="w-4 h-4 text-[#8A1538] shrink-0 mt-0.5" />
                <span>
                  <strong>Advisory:</strong> This analysis is an exploratory academic demonstration and does not constitute a clinical biopsy or definitive medical assessment.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
