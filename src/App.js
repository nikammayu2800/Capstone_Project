import logo from './logo.svg';
import './App.css';
import First_page from './frontend/components/first_page/First_page';

function App() {
  return (
    <div className="container App">
      <span className="has-background-primary-on-scheme">
        <First_page /> {/* Include the AnotherComponent */}
      </span>
    </div>
  );
}

export default App;
