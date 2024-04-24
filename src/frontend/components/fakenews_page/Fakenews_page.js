import "./Fakenews_page.css";
import { useState } from "react";
import { checkPosts, getPosts } from "./Fakenews_page_services";

function Fakenews_page() {
  const [postResult, setPostResult] = useState([]);
  const retrive_posts = async () => {
    try {
      const postData = await getPosts();
      postData.map((post, index) => {
        console.log("post", post);
        console.log("index", index);
      });
      // console.log(postData);
    } catch (error) {
      console.log(error);
    }
  };

  const check_posts = async () => {
    var inputValue = document.getElementById("newsInput").value;
    console.log(inputValue);
    try {
      const result = await checkPosts(inputValue);
      setPostResult(result);
      console.log(result);
    } catch (error) {
      console.log(error);
    }
  };

  const clear_input = async () => {
    document.getElementById("newsInput").value = "";
    setPostResult([]);
  };

  return (
    <>
      <div className="is-primary is-size-2 my-5">Fake News</div>
      <h3 className="my-3">Enter your News:</h3>
      <div className="field">
        <div className="control is-expanded">
          <div className="field has-addons">
            <input
              id="newsInput"
              className="input is-rounded my-3"
              type="text"
              placeholder="Enter News Here"
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
        <button class="button is-info" onClick={check_posts}>
          Check the news
        </button>
      </div>
      {postResult !== null && (
        <div className="result my-6">
          <h1
            className={`title ${
              postResult == "real" ? "real" : "fake"
            } animated`}
          >
            {postResult}
          </h1>
        </div>
      )}
    </>
  );
}

export default Fakenews_page;
