import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:5000"; // Update with your backend URL

const getPosts = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/get_posts`);
    return response.data;
  } catch (error) {
    console.error("Error fetching posts:", error.message);
    throw error;
  }
};

const checkPosts = async (inputValue) => {
  console.log("inputValue", inputValue);
  try {
    const response = await axios.post(`${API_BASE_URL}/check_post`, {
      inputValue: inputValue,
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching posts:", error.message);
    throw error;
  }
};

export { getPosts, checkPosts };
