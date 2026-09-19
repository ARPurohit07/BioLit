import { HashRouter, Route, Routes } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import { EvidenceViewerProvider } from "./components/EvidenceViewerContext";
import EvidenceViewer from "./components/EvidenceViewer";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Query from "./pages/Query";
import LiteratureReview from "./pages/LiteratureReview";
import Compare from "./pages/Compare";
import Evaluation from "./pages/Evaluation";
import Settings from "./pages/Settings";
import "./styles/app.css";

export default function App() {
  return (
    <HashRouter>
      <EvidenceViewerProvider>
        <div className="app-shell">
          <Sidebar />
          <main className="app-main">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/documents" element={<Documents />} />
              <Route path="/query" element={<Query />} />
              <Route path="/literature-review" element={<LiteratureReview />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
          <EvidenceViewer />
        </div>
      </EvidenceViewerProvider>
    </HashRouter>
  );
}
