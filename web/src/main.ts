import { mount } from "svelte";
import "katex/dist/katex.min.css";
import "./ui/style.css";
import App from "./ui/App.svelte";

mount(App, { target: document.getElementById("app")! });
