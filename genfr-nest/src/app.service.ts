import { Injectable } from "@nestjs/common";

@Injectable()
export class AppService {
  getHello(): {
    title: string;
    authors: string;
  } {
    return {
      title: "Generator Frajdy - pixel art",
      authors: "Team generator frajdy",
    };
  }
}
