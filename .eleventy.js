// Required Plugins
const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const markdownIt = require("markdown-it");
const mdObsidianCallouts = require("markdown-it-obsidian-callouts");

module.exports = function(eleventyConfig) {

  // Copy the 'css' folder to the final build
  eleventyConfig.addPassthroughCopy("css");

  eleventyConfig.addFilter("postDate", (dateObj) => {
      // Locale 'en-US'
      return dateObj.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    });

  // Date Formatting Filter for ISO Strings
  eleventyConfig.addFilter("isoDate", (dateObj) => {
      return dateObj.toISOString();
    });
  
  // Syntax Highlighting Plugin
  eleventyConfig.addPlugin(syntaxHighlight);

  // Custom Markdown Formatter
  let mdLib = markdownIt({
    html: true // Enable HTML tags in Markdown
  }).use(mdObsidianCallouts);

  eleventyConfig.setLibrary("md", mdLib);

  // Main Configuration
  return {
    dir: {
      input: ".",               // Top-level folder : Content
      includes: "_includes",    // Templates 
      output: "_site"           // Built site
    },
    // Process .html files as templates and allow Markdown.
    htmlTemplateEngine: "njk",
    markdownTemplateEngine: "njk"
  };
};