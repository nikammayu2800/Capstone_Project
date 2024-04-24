import "./First_page.css";
import React, { useState } from "react";
import Fakenews from "../../Images/FN.jpeg";
import { useNavigate } from "react-router-dom";
import Cyberbullying from "../../Images/CB_1.jpeg";
import { getComments, getPosts } from "./First_page_service";

function First_page() {
  const navigate = useNavigate();

  const handleCyberbullyingClick = () => {
    // Navigate to the cyberbullying page
    navigate("/cyberbullying");
  };

  const handleFakeNewsClick = () => {
    // Navigate to the fake news page
    navigate("/fake-news");
  };

  return (
    <>
      <div className="is-primary is-size-2 my-5">
        Fake News and Cyberbullying Detection
      </div>
      {/* Cyberbullying Container */}
      <div className="fixed-grid has-2-cols">
        <div className="grid">
          <div className="my-5 mx-5">
            <img src={Cyberbullying} alt="Left Side Image" />
          </div>
          <div className="content">
            <h3 class="title">Cyberbullying</h3>
            <p className="justify-content">
              Cyberbullying is the use of digital communication tools to harass,
              intimidate, or harm others. It often takes place through social
              media, messaging apps, or online forums. Cyberbullies may target
              their victims with hurtful messages, rumors, or threats, leading
              to emotional distress and even serious consequences such as
              depression and self-harm. It's a pervasive issue that requires
              proactive measures to combat and protect individuals from online
              abuse.
            </p>
            <button class="button is-info" onClick={handleCyberbullyingClick}>
              Check It
            </button>
          </div>
        </div>
      </div>
      {/* FakeNews Container */}
      <div className="fixed-grid has-2-cols">
        <div className="grid">
          <div className="content">
            <h3 class="title">FakeNews</h3>
            <p className="justify-content">
              Fake news refers to false or misleading information presented as
              factual news. It can spread rapidly through various media
              platforms, causing confusion and influencing public opinion. Fake
              news often aims to deceive or manipulate readers for political,
              financial, or other reasons. Its proliferation underscores the
              importance of critical thinking and fact-checking in the digital
              age. Combating fake news requires vigilance from both consumers
              and content creators to uphold the integrity of information
              dissemination.
            </p>
            <button class="button is-info" onClick={handleFakeNewsClick}>
              Check It
            </button>
          </div>
          <div>
            <img src={Fakenews} alt="Left Side Image" />
          </div>
        </div>
      </div>
    </>
  );
}

export default First_page;
