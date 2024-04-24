import { useState } from "react";
import { checkComments, getComments } from "./Cyberbullying_page_services";
import "./Cyberbullying_page.css";

function Cyberbullying_page() {
  const [commentsData, setCommentsData] = useState([]);
  const [commentResult, setPostResult] = useState([]);

  const retrive_comments = async () => {
    try {
      const result = await getComments();
      setCommentsData(result);
      console.log(result);
    } catch (error) {
      console.log(error);
    }
  };

  const check_comment = async () => {
    var inputValue = document.getElementById("commentInput").value;
    console.log("inputValue", inputValue);
    try {
      const result = await checkComments(inputValue, commentsData);
      setPostResult(result);
      console.log(result);
    } catch (error) {
      console.log(error);
    }
  };

  const clear_input = async () => {
    document.getElementById("commentInput").value = "";
    setPostResult([]);
  };

  // Function to render table rows for comments
  const renderComments = () => {
    return commentsData.map((comment, index) => (
      <tr key={index}>
        <th>{index + 1}</th>
        <td>{comment}</td>
      </tr>
    ));
  };

  return (
    <>
      <div className="is-primary is-size-2 my-5">Cyberbullying</div>
      <div>
        <button class="button" onClick={retrive_comments}>
          retrive comments
        </button>
      </div>

      {commentsData.length > 0 && (
        <div
          class="table-container mt-6"
          style={{ maxHeight: "300px", overflowY: "auto" }}
        >
          <table class="table">
            <thead>
              <tr>
                <th>Sr. No.</th>
                <th>Comments</th>
              </tr>
            </thead>
            <tbody>{renderComments()}</tbody>
          </table>
        </div>
      )}
      <h3 className="my-3">Enter your comment:</h3>
      <div className="field">
        <div className="control is-expanded">
          <div className="field has-addons">
            <input
              id="commentInput"
              className="input is-rounded my-3"
              type="text"
              placeholder="Enter Comment Here"
            />
            <button className="button clear-button ml-1" onClick={clear_input}>
              <span className="icon is-small is-right clickable">
                <i className="fas fa-times"></i>
              </span>
            </button>
          </div>
        </div>
      </div>
      <div>
        <button class="button is-info" onClick={check_comment}>
          Check the news
        </button>
      </div>
      {commentResult !== null && (
        <div className="result my-6">
          <h1
            className={`title ${
              commentResult == "Non-Cyberbullying"
                ? "non-cyberbullying"
                : "cyberbullying"
            } animated`}
          >
            {commentResult}
          </h1>
        </div>
      )}
    </>
  );
}

export default Cyberbullying_page;
