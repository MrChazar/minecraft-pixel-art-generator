import { MailerService } from '@nestjs-modules/mailer';
import { Injectable } from '@nestjs/common';

@Injectable()
export class EmailService {
    constructor(private readonly mailService: MailerService) {}

    async sendTest(email:string){
        await this.mailService.sendMail({
            to:email,
            subject:"hello its a test",
            text:"this works",
            html: `<p>Hello ${email}</p>`,
            from: '"Generator Frajdy" <noreply@yourdomain.com>',
        })
    }
}
