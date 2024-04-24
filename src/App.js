import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import First_page from "./frontend/components/first_page/First_page";
import Cyberbullying_page from "./frontend/components/cyberbullying_page/Cyberbullying_page";
import Fakenews_page from "./frontend/components/fakenews_page/Fakenews_page";

function App() {
  return (
    <>
      <div className="container App">
        <span className="has-background-primary-on-scheme">
          <Routes>
            <Route exact path="/" element={<First_page />} />
            <Route path="/cyberbullying" element={<Cyberbullying_page />} />
            <Route path="/fake-news" element={<Fakenews_page />} />
          </Routes>
        </span>
      </div>
      {/* <First_page /> Include the AnotherComponent */}
    </>
  );
}

export default App;
