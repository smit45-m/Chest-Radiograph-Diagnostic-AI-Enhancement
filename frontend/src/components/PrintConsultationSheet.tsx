import React from "react";
import { formatPathology } from "../data/clinicalCatalog";
import type {
  HeatmapData,
  PatientProfile,
  PredictionResponse,
  ReportData,
} from "../types/clinical";

interface PrintConsultationSheetProps {
  studyName: string;
  imageUrl: string | null;
  predictionData: PredictionResponse | null;
  activeHeatmap: HeatmapData | null;
  report: ReportData | null;
  patient: PatientProfile;
  threshold: number;
  physicianName: string;
  physicianCreds: string;
  physicianNotes: string;
  signedAt: string | null;
}

export const PrintConsultationSheet: React.FC<PrintConsultationSheetProps> = ({
  studyName,
  imageUrl,
  predictionData,
  activeHeatmap,
  report,
  patient,
  threshold,
  physicianName,
  physicianCreds,
  physicianNotes,
  signedAt,
}) => {
  if (!predictionData || !report) return null;

  const positives = predictionData.predictions.filter((p) => p.score >= threshold);

  return (
    <div className="print-only mx-auto max-w-4xl bg-white p-8 text-slate-900">
      <div className="flex items-start justify-between border-b-2 border-slate-900 pb-4">
        <div>
          <div className="font-mono-tech text-xs tracking-widest uppercase text-teal-800">
            CXR·01 · THORACIC RADIOLOGY CONSULTATION SHEET
          </div>
          <h1 className="mt-1 text-2xl font-bold">Chest Radiograph AI-Assisted Report</h1>
          <div className="mt-1 text-xs text-slate-600">
            Study: {studyName} · Generated {new Date().toLocaleString()}
          </div>
        </div>
        <div className="text-right font-mono-tech text-xs">
          <div className="font-bold">{report.triage_level || "ROUTINE"}</div>
          <div>CTR: {predictionData.anatomy?.ctr?.toFixed(3) ?? "N/A"}</div>
          <div>Provider: {report.source}</div>
        </div>
      </div>

      {/* Patient Bar */}
      <div className="mt-4 grid grid-cols-3 gap-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs">
        <div>
          <span className="font-semibold">Patient:</span> {patient.age ? `${patient.age}y ` : ""}
          {patient.sex}
        </div>
        <div>
          <span className="font-semibold">Setting:</span> {patient.setting}
        </div>
        <div>
          <span className="font-semibold">Smoking:</span> {patient.smoking}
        </div>
        <div className="col-span-3">
          <span className="font-semibold">Indication:</span> {patient.indication}
        </div>
      </div>

      {/* Key Images */}
      <div className="mt-5 grid grid-cols-2 gap-4">
        {imageUrl && (
          <div className="rounded-lg border border-slate-200 p-2">
            <img src={imageUrl} alt="Frontal CXR" className="h-56 w-full object-contain bg-black" />
            <div className="mt-1 text-center font-mono-tech text-[10px] text-slate-600">
              Source Frontal Radiograph
            </div>
          </div>
        )}
        {activeHeatmap?.image_base64 && (
          <div className="rounded-lg border border-slate-200 p-2">
            <img
              src={`data:image/png;base64,${activeHeatmap.image_base64}`}
              alt="Grad-CAM++ Map"
              className="h-56 w-full object-contain bg-black"
            />
            <div className="mt-1 text-center font-mono-tech text-[10px] text-slate-600">
              Grad-CAM++ · {formatPathology(activeHeatmap.pathology)}
            </div>
          </div>
        )}
      </div>

      {/* Detected Positives */}
      <div className="mt-5">
        <h2 className="font-mono-tech text-xs font-bold tracking-wider uppercase text-slate-700">
          Quantitative Ensemble Findings (Cutoff {Math.round(threshold * 100)}%)
        </h2>
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {positives.length > 0 ? (
            positives.map((p) => (
              <span
                key={p.name}
                className="rounded border border-slate-300 bg-slate-100 px-2.5 py-1 font-mono-tech"
              >
                {formatPathology(p.name)}: {(p.score * 100).toFixed(1)}%
              </span>
            ))
          ) : (
            <span className="text-slate-600">No thoracic pathology exceeded operating threshold.</span>
          )}
        </div>
      </div>

      {/* Impression & Findings */}
      <div className="mt-5 space-y-4 text-sm leading-relaxed">
        <div>
          <h2 className="font-mono-tech text-xs font-bold tracking-wider uppercase text-slate-700">
            Impression
          </h2>
          <p className="mt-1 whitespace-pre-line font-semibold">{report.impression}</p>
        </div>

        <div>
          <h2 className="font-mono-tech text-xs font-bold tracking-wider uppercase text-slate-700">
            Findings
          </h2>
          <p className="mt-1 whitespace-pre-line">{report.findings}</p>
        </div>

        {report.recommendations && (
          <div>
            <h2 className="font-mono-tech text-xs font-bold tracking-wider uppercase text-slate-700">
              Recommendations
            </h2>
            <p className="mt-1 whitespace-pre-line">{report.recommendations}</p>
          </div>
        )}
      </div>

      {/* Physician Attestation */}
      <div className="mt-6 border-t border-slate-300 pt-4 text-xs">
        <div className="font-semibold">
          Attending Attestation: {physicianName} {physicianCreds ? `(${physicianCreds})` : ""}
        </div>
        {physicianNotes && <p className="mt-1 text-slate-700">Addendum: {physicianNotes}</p>}
        <div className="mt-1 text-slate-500">
          {signedAt ? `Digitally attested on ${signedAt}` : "Draft — pending attending signature"}
        </div>
      </div>
    </div>
  );
};
