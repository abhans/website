---
layout: base.html
title: Blog
---

# Welcome to the Treasury

Here you can find all my notes, studies, and personal thoughts.

## Posts

<ul class="postlist">
  {%- for myPost in collections.post | reverse -%}
    <li class="postlistItem">
      <a href="{{ myPost.url }}">
        {{ myPost.data.title }}
      </a>
      <time class="postlistDate" datetime="{{ myPost.date | isoDate }}">
        {{ myPost.date | postDate }}
      </time>
    </li>
  {%- endfor -%}
</ul>