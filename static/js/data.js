/* Generated from the RoboWits repo (README benchmark tables + gs_gym/envs/robowits).
   Regenerate rather than hand-edit. */
window.RW = {
 "tasks": [
  {
   "id": "01",
   "name": "Align Blocks",
   "instruction": "Align the three cubes perfectly in a straight line.",
   "mutations": 6,
   "video": "01.mp4",
   "altVideo": null,
   "poster": "01.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      14.0,
      0.61
     ],
     "mut": [
      10.0,
      0.58
     ]
    },
    "pi0": {
     "seed": [
      14.0,
      0.63
     ],
     "mut": [
      20.8,
      0.62
     ]
    },
    "pi05": {
     "seed": [
      20.0,
      0.63
     ],
     "mut": [
      15.0,
      0.63
     ]
    }
   }
  },
  {
   "id": "02",
   "name": "Retrieve Cube",
   "instruction": "Retrieve the cube from the container and place it on the green target area.",
   "mutations": 7,
   "video": "02.mp4",
   "altVideo": "02_2.mp4",
   "poster": "02.webp",
   "core10": true,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      2.0,
      0.29
     ],
     "mut": [
      0.0,
      0.32
     ]
    },
    "pi0": {
     "seed": [
      6.0,
      0.3
     ],
     "mut": [
      0.8,
      0.31
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.28
     ],
     "mut": [
      1.7,
      0.32
     ]
    }
   }
  },
  {
   "id": "03",
   "name": "Gap Retrieve",
   "instruction": "Place the lemon on the green target area.",
   "mutations": 5,
   "video": "03.mp4",
   "altVideo": null,
   "poster": "03.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      2.0,
      0.51
     ],
     "mut": [
      0.0,
      0.43
     ]
    },
    "pi0": {
     "seed": [
      18.0,
      0.65
     ],
     "mut": [
      5.0,
      0.46
     ]
    },
    "pi05": {
     "seed": [
      20.0,
      0.66
     ],
     "mut": [
      6.0,
      0.49
     ]
    }
   }
  },
  {
   "id": "04",
   "name": "Pinch Card",
   "instruction": "Pick the bank card up and place it on the green target area.",
   "mutations": 5,
   "video": "04.mp4",
   "altVideo": null,
   "poster": "04.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.41
     ],
     "mut": [
      0.0,
      0.53
     ]
    },
    "pi0": {
     "seed": [
      10.0,
      0.63
     ],
     "mut": [
      9.0,
      0.62
     ]
    },
    "pi05": {
     "seed": [
      2.0,
      0.54
     ],
     "mut": [
      7.0,
      0.59
     ]
    }
   }
  },
  {
   "id": "05",
   "name": "Roll Up Ball",
   "instruction": "Move the box into the basket.",
   "mutations": 8,
   "video": "05.mp4",
   "altVideo": null,
   "poster": "05.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.36
     ],
     "mut": [
      0.0,
      0.25
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.42
     ],
     "mut": [
      0.0,
      0.31
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.5
     ],
     "mut": [
      0.0,
      0.31
     ]
    }
   }
  },
  {
   "id": "06",
   "name": "Dominos",
   "instruction": "Push the red block onto the green target area.",
   "mutations": 6,
   "video": "06.mp4",
   "altVideo": null,
   "poster": "06.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      52.0,
      0.87
     ],
     "mut": [
      38.3,
      0.86
     ]
    },
    "pi0": {
     "seed": [
      86.0,
      0.97
     ],
     "mut": [
      57.5,
      0.87
     ]
    },
    "pi05": {
     "seed": [
      74.0,
      0.95
     ],
     "mut": [
      52.5,
      0.84
     ]
    }
   }
  },
  {
   "id": "07",
   "name": "Stand Pages",
   "instruction": "Stand up the two pages stably.",
   "mutations": 2,
   "video": "07.mp4",
   "altVideo": null,
   "poster": "07.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.24
     ],
     "mut": [
      0.0,
      0.16
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.29
     ],
     "mut": [
      0.0,
      0.29
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.29
     ],
     "mut": [
      0.0,
      0.13
     ]
    }
   }
  },
  {
   "id": "08",
   "name": "Round Dough Sheet",
   "instruction": "Flatten the dough ball and make a perfect round sheet.",
   "mutations": 6,
   "video": null,
   "altVideo": null,
   "poster": "08.webp",
   "core10": false,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.52
     ],
     "mut": [
      0.0,
      0.52
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.58
     ],
     "mut": [
      0.0,
      0.58
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.5
     ],
     "mut": [
      0.0,
      0.52
     ]
    }
   }
  },
  {
   "id": "09",
   "name": "Hold Cup",
   "instruction": "Put the cup on the green target area on the slope without sliding down.",
   "mutations": 5,
   "video": "09.mp4",
   "altVideo": null,
   "poster": "09.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.53
     ],
     "mut": [
      0.0,
      0.39
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.66
     ],
     "mut": [
      0.0,
      0.45
     ]
    },
    "pi05": {
     "seed": [
      2.0,
      0.64
     ],
     "mut": [
      0.0,
      0.46
     ]
    }
   }
  },
  {
   "id": "10",
   "name": "Collect Screws",
   "instruction": "Collect the screw into the container.",
   "mutations": 5,
   "video": "10.mp4",
   "altVideo": null,
   "poster": "10.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.26
     ],
     "mut": [
      0.0,
      0.31
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.29
     ],
     "mut": [
      0.0,
      0.32
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.26
     ],
     "mut": [
      0.0,
      0.33
     ]
    }
   }
  },
  {
   "id": "11",
   "name": "Place Tall Box",
   "instruction": "Move the rectangular box onto the green target area.",
   "mutations": 5,
   "video": "11.mp4",
   "altVideo": null,
   "poster": "11.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.61
     ],
     "mut": [
      3.0,
      0.68
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.79
     ],
     "mut": [
      1.0,
      0.69
     ]
    },
    "pi05": {
     "seed": [
      6.0,
      0.83
     ],
     "mut": [
      3.0,
      0.7
     ]
    }
   }
  },
  {
   "id": "12",
   "name": "Ball Into Bottle",
   "instruction": "Transfer the ball from the blue container into the red bottle.",
   "mutations": 6,
   "video": "12.mp4",
   "altVideo": null,
   "poster": "12.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.44
     ],
     "mut": [
      0.0,
      0.32
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.49
     ],
     "mut": [
      0.0,
      0.37
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.49
     ],
     "mut": [
      0.0,
      0.35
     ]
    }
   }
  },
  {
   "id": "13",
   "name": "Cover With Lid",
   "instruction": "Cover the pot precisely using one lid.",
   "mutations": 6,
   "video": "13.mp4",
   "altVideo": null,
   "poster": "13.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.48
     ],
     "mut": [
      0.0,
      0.37
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.53
     ],
     "mut": [
      0.0,
      0.43
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.58
     ],
     "mut": [
      0.0,
      0.48
     ]
    }
   }
  },
  {
   "id": "14",
   "name": "Stack Cubes",
   "instruction": "Build a stable two-layer stack so the red cube's center is above the green colored dot.",
   "mutations": 6,
   "video": "14.mp4",
   "altVideo": null,
   "poster": "14.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.36
     ],
     "mut": [
      0.0,
      0.29
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.4
     ],
     "mut": [
      0.0,
      0.31
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.37
     ],
     "mut": [
      0.0,
      0.3
     ]
    }
   }
  },
  {
   "id": "15",
   "name": "Separate Marbles & Sand",
   "instruction": "Remove sand from the jar and keep the marbles in the jar.",
   "mutations": 4,
   "video": null,
   "altVideo": null,
   "poster": "15.webp",
   "core10": false,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      16.0,
      0.43
     ],
     "mut": [
      11.2,
      0.28
     ]
    },
    "pi0": {
     "seed": [
      36.0,
      0.81
     ],
     "mut": [
      30.0,
      0.7
     ]
    },
    "pi05": {
     "seed": [
      42.0,
      0.8
     ],
     "mut": [
      17.5,
      0.53
     ]
    }
   }
  },
  {
   "id": "16",
   "name": "Stand Bulb",
   "instruction": "Stand the tall object with curved bottom stably within the green target area.",
   "mutations": 7,
   "video": "16.mp4",
   "altVideo": null,
   "poster": "16.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.54
     ],
     "mut": [
      0.0,
      0.41
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.55
     ],
     "mut": [
      0.0,
      0.42
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.55
     ],
     "mut": [
      0.0,
      0.42
     ]
    }
   }
  },
  {
   "id": "17",
   "name": "Ball Onto Tower",
   "instruction": "Place the ball stably on top of the tower.",
   "mutations": 8,
   "video": "17.mp4",
   "altVideo": null,
   "poster": "17.webp",
   "core10": true,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.55
     ],
     "mut": [
      0.0,
      0.48
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.62
     ],
     "mut": [
      0.0,
      0.54
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.67
     ],
     "mut": [
      0.0,
      0.58
     ]
    }
   }
  },
  {
   "id": "18",
   "name": "Cylinder Through Hole",
   "instruction": "Insert the yellow rod into one of the round holes in the white block so it stands upright in the hole.",
   "mutations": 6,
   "video": "18.mp4",
   "altVideo": null,
   "poster": "18.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.3
     ],
     "mut": [
      0.0,
      0.26
     ]
    },
    "pi0": {
     "seed": [
      2.0,
      0.57
     ],
     "mut": [
      0.8,
      0.38
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.39
     ],
     "mut": [
      0.8,
      0.41
     ]
    }
   }
  },
  {
   "id": "19",
   "name": "Stack Bowls",
   "instruction": "Nest the smaller bowl concentrically inside the larger one.",
   "mutations": 6,
   "video": "19.mp4",
   "altVideo": null,
   "poster": "19.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      2.0,
      0.53
     ],
     "mut": [
      0.0,
      0.3
     ]
    },
    "pi0": {
     "seed": [
      24.0,
      0.68
     ],
     "mut": [
      0.8,
      0.4
     ]
    },
    "pi05": {
     "seed": [
      26.0,
      0.73
     ],
     "mut": [
      0.8,
      0.35
     ]
    }
   }
  },
  {
   "id": "20",
   "name": "Ball Into Jar",
   "instruction": "Put the foam ball fully into the jar.",
   "mutations": 5,
   "video": null,
   "altVideo": null,
   "poster": "20.webp",
   "core10": false,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      6.0,
      0.55
     ],
     "mut": [
      2.0,
      0.3
     ]
    },
    "pi0": {
     "seed": [
      6.0,
      0.59
     ],
     "mut": [
      2.0,
      0.29
     ]
    },
    "pi05": {
     "seed": [
      6.0,
      0.55
     ],
     "mut": [
      2.0,
      0.31
     ]
    }
   }
  },
  {
   "id": "21",
   "name": "Seal Colander",
   "instruction": "Make the container hold water for a short time.",
   "mutations": 5,
   "video": null,
   "altVideo": null,
   "poster": "21.webp",
   "core10": false,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.15
     ],
     "mut": [
      0.0,
      0.16
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.18
     ],
     "mut": [
      0.0,
      0.18
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.17
     ],
     "mut": [
      0.0,
      0.17
     ]
    }
   }
  },
  {
   "id": "22",
   "name": "Stabilize Egg",
   "instruction": "Make the ellipsoid object stand upright in the bowl without falling.",
   "mutations": 5,
   "video": null,
   "altVideo": null,
   "poster": "22.webp",
   "core10": false,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.34
     ],
     "mut": [
      0.0,
      0.36
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.47
     ],
     "mut": [
      0.0,
      0.42
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.35
     ],
     "mut": [
      0.0,
      0.38
     ]
    }
   }
  },
  {
   "id": "23",
   "name": "Place Book",
   "instruction": "Shelve the book upright without knocking over the other books.",
   "mutations": 5,
   "video": "23.mp4",
   "altVideo": null,
   "poster": "23.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      4.0,
      0.21
     ],
     "mut": [
      3.0,
      0.25
     ]
    },
    "pi0": {
     "seed": [
      2.0,
      0.3
     ],
     "mut": [
      3.0,
      0.4
     ]
    },
    "pi05": {
     "seed": [
      2.0,
      0.29
     ],
     "mut": [
      26.0,
      0.54
     ]
    }
   }
  },
  {
   "id": "24",
   "name": "Raise Platform",
   "instruction": "Create an elevated platform and place the book on it.",
   "mutations": 4,
   "video": "24.mp4",
   "altVideo": null,
   "poster": "24.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      4.0,
      0.29
     ],
     "mut": [
      0.0,
      0.29
     ]
    },
    "pi0": {
     "seed": [
      20.0,
      0.52
     ],
     "mut": [
      0.0,
      0.32
     ]
    },
    "pi05": {
     "seed": [
      18.0,
      0.49
     ],
     "mut": [
      0.0,
      0.35
     ]
    }
   }
  },
  {
   "id": "25",
   "name": "Water Into Mug",
   "instruction": "Collect water in the mug without moving the pitcher.",
   "mutations": 7,
   "video": null,
   "altVideo": null,
   "poster": "25.webp",
   "core10": true,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      2.0,
      0.36
     ],
     "mut": [
      0.7,
      0.43
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.36
     ],
     "mut": [
      0.0,
      0.45
     ]
    },
    "pi05": {
     "seed": [
      4.0,
      0.35
     ],
     "mut": [
      0.0,
      0.47
     ]
    }
   }
  },
  {
   "id": "26",
   "name": "Align Chopsticks",
   "instruction": "Make all six chopsticks point in the same direction, with their small thin ends all facing the same way.",
   "mutations": 5,
   "video": "26.mp4",
   "altVideo": null,
   "poster": "26.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.0
     ],
     "mut": [
      0.0,
      0.06
     ]
    },
    "pi0": {
     "seed": [
      4.0,
      0.06
     ],
     "mut": [
      0.0,
      0.08
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.01
     ],
     "mut": [
      0.0,
      0.1
     ]
    }
   }
  },
  {
   "id": "27",
   "name": "Retrieve Roll",
   "instruction": "Place the roll on the green target area.",
   "mutations": 7,
   "video": "27.mp4",
   "altVideo": null,
   "poster": "27.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.2
     ],
     "mut": [
      0.0,
      0.21
     ]
    },
    "pi0": {
     "seed": [
      2.0,
      0.22
     ],
     "mut": [
      1.4,
      0.22
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.21
     ],
     "mut": [
      0.0,
      0.21
     ]
    }
   }
  },
  {
   "id": "28",
   "name": "Move Cube",
   "instruction": "Move any cube beyond the goal line.",
   "mutations": 7,
   "video": "28.mp4",
   "altVideo": null,
   "poster": "28.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      10.0,
      0.59
     ],
     "mut": [
      0.0,
      0.45
     ]
    },
    "pi0": {
     "seed": [
      24.0,
      0.69
     ],
     "mut": [
      0.0,
      0.49
     ]
    },
    "pi05": {
     "seed": [
      12.0,
      0.66
     ],
     "mut": [
      0.7,
      0.5
     ]
    }
   }
  },
  {
   "id": "29",
   "name": "Balance Board",
   "instruction": "Place the board onto the wall and keep it balanced.",
   "mutations": 6,
   "video": "29.mp4",
   "altVideo": null,
   "poster": "29.webp",
   "core10": false,
   "soft": false,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.26
     ],
     "mut": [
      0.0,
      0.34
     ]
    },
    "pi0": {
     "seed": [
      2.0,
      0.44
     ],
     "mut": [
      0.0,
      0.35
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.35
     ],
     "mut": [
      0.0,
      0.38
     ]
    }
   }
  },
  {
   "id": "30",
   "name": "Differentiate Cubes",
   "instruction": "Place the wooden cube to the green target area. Keep the metal cube inside the mug.",
   "mutations": 5,
   "video": null,
   "altVideo": null,
   "poster": "30.webp",
   "core10": false,
   "soft": true,
   "results": {
    "act": {
     "seed": [
      0.0,
      0.27
     ],
     "mut": [
      0.0,
      0.34
     ]
    },
    "pi0": {
     "seed": [
      0.0,
      0.28
     ],
     "mut": [
      0.0,
      0.4
     ]
    },
    "pi05": {
     "seed": [
      0.0,
      0.3
     ],
     "mut": [
      1.0,
      0.37
     ]
    }
   }
  }
 ],
 "averages": {
  "rw10": {
   "act": {
    "seed": [
     7.2,
     0.52
    ],
    "mut": [
     4.9,
     0.48
    ]
   },
   "pi0": {
    "seed": [
     13.4,
     0.59
    ],
    "mut": [
     9.3,
     0.52
    ]
   },
   "pi05": {
    "seed": [
     12.2,
     0.59
    ],
    "mut": [
     8.2,
     0.53
    ]
   }
  },
  "full": {
   "act": {
    "seed": [
     3.8,
     0.4
    ],
    "mut": [
     2.3,
     0.36
    ]
   },
   "pi0": {
    "seed": [
     8.5,
     0.5
    ],
    "mut": [
     4.4,
     0.42
    ]
   },
   "pi05": {
    "seed": [
     7.8,
     0.48
    ],
    "mut": [
     4.5,
     0.42
    ]
   }
  }
 },
 "mutationPairs": [
  {
   "task": "01",
   "name": "Align Blocks",
   "instruction": "Align the three cubes perfectly in a straight line.",
   "clips": [
    {
     "v": "01.mp4",
     "label": "Seed task",
     "kind": "seed",
     "note": "A rigid ruler is available as a straightedge."
    },
    {
     "v": "01_mut1.mp4",
     "label": "Add distractors",
     "kind": "add",
     "note": "Irrelevant objects crowd the workspace; the ruler still works."
    },
    {
     "v": "01_mut2.mp4",
     "label": "Fix two blocks",
     "kind": "fix",
     "note": "Two cubes are welded to the table — only the third may move."
    }
   ]
  },
  {
   "task": "04",
   "name": "Pinch Card",
   "instruction": "Pick the bank card up and place it on the green target area.",
   "clips": [
    {
     "v": "04.mp4",
     "label": "Seed task",
     "kind": "seed",
     "note": "A flat card must be lifted off a flat table."
    },
    {
     "v": "04_mut1.mp4",
     "label": "Add distractors",
     "kind": "add",
     "note": "Extra objects are added around the card."
    },
    {
     "v": "04_mut2.mp4",
     "label": "Add an obstacle",
     "kind": "add",
     "note": "An obstacle blocks the straight-line approach."
    }
   ]
  },
  {
   "task": "06",
   "name": "Dominos",
   "instruction": "Push the red block onto the green target area.",
   "clips": [
    {
     "v": "06.mp4",
     "label": "Seed task",
     "kind": "seed",
     "note": "A chain of blocks transmits the push to the target."
    },
    {
     "v": "06_mut1.mp4",
     "label": "Add distractors",
     "kind": "add",
     "note": "Irrelevant objects are placed near the chain."
    },
    {
     "v": "06_mut2.mp4",
     "label": "Replace the pusher",
     "kind": "replace",
     "note": "The pusher is swapped for an object with different affordances."
    }
   ]
  },
  {
   "task": "09",
   "name": "Hold Cup",
   "instruction": "Put the cup on the green target area on the slope without sliding down.",
   "clips": [
    {
     "v": "09.mp4",
     "label": "Seed task",
     "kind": "seed",
     "note": "A coaster keeps the cup from sliding down the slope."
    },
    {
     "v": "09_mut1.mp4",
     "label": "Replace the coaster",
     "kind": "replace",
     "note": "The coaster is swapped for a differently shaped support."
    },
    {
     "v": "09_mut2.mp4",
     "label": "Fix the coaster",
     "kind": "fix",
     "note": "The coaster is immobilised, so it can no longer be repositioned."
    }
   ]
  }
 ]
};
